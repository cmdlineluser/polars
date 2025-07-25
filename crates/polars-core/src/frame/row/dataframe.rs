use super::*;

impl DataFrame {
    /// Get a row from a [`DataFrame`]. Use of this is discouraged as it will likely be slow.
    pub fn get_row(&self, idx: usize) -> PolarsResult<Row<'_>> {
        let values = self
            .materialized_column_iter()
            .map(|s| s.get(idx))
            .collect::<PolarsResult<Vec<_>>>()?;
        Ok(Row(values))
    }

    /// Amortize allocations by reusing a row.
    /// The caller is responsible to make sure that the row has at least the capacity for the number
    /// of columns in the [`DataFrame`]
    pub fn get_row_amortized<'a>(&'a self, idx: usize, row: &mut Row<'a>) -> PolarsResult<()> {
        for (s, any_val) in self.materialized_column_iter().zip(&mut row.0) {
            *any_val = s.get(idx)?;
        }
        Ok(())
    }

    /// Amortize allocations by reusing a row.
    /// The caller is responsible to make sure that the row has at least the capacity for the number
    /// of columns in the [`DataFrame`]
    ///
    /// # Safety
    /// Does not do any bounds checking.
    #[inline]
    pub unsafe fn get_row_amortized_unchecked<'a>(&'a self, idx: usize, row: &mut Row<'a>) {
        self.materialized_column_iter()
            .zip(&mut row.0)
            .for_each(|(s, any_val)| {
                *any_val = s.get_unchecked(idx);
            });
    }

    /// Create a new [`DataFrame`] from rows.
    ///
    /// This should only be used when you have row wise data, as this is a lot slower
    /// than creating the [`Series`] in a columnar fashion
    pub fn from_rows_and_schema(rows: &[Row], schema: &Schema) -> PolarsResult<Self> {
        Self::from_rows_iter_and_schema(rows.iter(), schema)
    }

    /// Create a new [`DataFrame`] from an iterator over rows.
    ///
    /// This should only be used when you have row wise data, as this is a lot slower
    /// than creating the [`Series`] in a columnar fashion.
    pub fn from_rows_iter_and_schema<'a, I>(mut rows: I, schema: &Schema) -> PolarsResult<Self>
    where
        I: Iterator<Item = &'a Row<'a>>,
    {
        if schema.is_empty() {
            let height = rows.count();
            let columns = Vec::new();
            return Ok(unsafe { DataFrame::new_no_checks(height, columns) });
        }

        let capacity = rows.size_hint().0;

        let mut buffers: Vec<_> = schema
            .iter_values()
            .map(|dtype| {
                let buf: AnyValueBuffer = (dtype, capacity).into();
                buf
            })
            .collect();

        let mut expected_len = 0;
        rows.try_for_each::<_, PolarsResult<()>>(|row| {
            expected_len += 1;
            for (value, buf) in row.0.iter().zip(&mut buffers) {
                buf.add_fallible(value)?
            }
            Ok(())
        })?;
        let v = buffers
            .into_iter()
            .zip(schema.iter_names())
            .map(|(b, name)| {
                let mut c = b.into_series().into_column();
                // if the schema adds a column not in the rows, we
                // fill it with nulls
                if c.is_empty() {
                    Column::full_null(name.clone(), expected_len, c.dtype())
                } else {
                    c.rename(name.clone());
                    c
                }
            })
            .collect();
        DataFrame::new(v)
    }

    /// Create a new [`DataFrame`] from an iterator over rows. This should only be used when you have row wise data,
    /// as this is a lot slower than creating the [`Series`] in a columnar fashion
    pub fn try_from_rows_iter_and_schema<'a, I>(mut rows: I, schema: &Schema) -> PolarsResult<Self>
    where
        I: Iterator<Item = PolarsResult<&'a Row<'a>>>,
    {
        let capacity = rows.size_hint().0;

        let mut buffers: Vec<_> = schema
            .iter_values()
            .map(|dtype| {
                let buf: AnyValueBuffer = (dtype, capacity).into();
                buf
            })
            .collect();

        let mut expected_len = 0;
        rows.try_for_each::<_, PolarsResult<()>>(|row| {
            expected_len += 1;
            for (value, buf) in row?.0.iter().zip(&mut buffers) {
                buf.add_fallible(value)?
            }
            Ok(())
        })?;
        let v = buffers
            .into_iter()
            .zip(schema.iter_names())
            .map(|(b, name)| {
                let mut c = b.into_series().into_column();
                // if the schema adds a column not in the rows, we
                // fill it with nulls
                if c.is_empty() {
                    Column::full_null(name.clone(), expected_len, c.dtype())
                } else {
                    c.rename(name.clone());
                    c
                }
            })
            .collect();
        DataFrame::new(v)
    }

    /// Create a new [`DataFrame`] from rows. This should only be used when you have row wise data,
    /// as this is a lot slower than creating the [`Series`] in a columnar fashion
    pub fn from_rows(rows: &[Row]) -> PolarsResult<Self> {
        let schema = rows_to_schema_first_non_null(rows, Some(50))?;
        let has_nulls = schema
            .iter_values()
            .any(|dtype| matches!(dtype, DataType::Null));
        polars_ensure!(
            !has_nulls, ComputeError: "unable to infer row types because of null values"
        );
        Self::from_rows_and_schema(rows, &schema)
    }
}
