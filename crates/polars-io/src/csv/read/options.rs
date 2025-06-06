#![allow(unsafe_op_in_unsafe_fn)]
use std::path::PathBuf;
use std::sync::Arc;

use polars_core::datatypes::{DataType, Field};
use polars_core::schema::{Schema, SchemaRef};
use polars_error::PolarsResult;
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::RowIndex;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct CsvReadOptions {
    pub path: Option<PathBuf>,
    // Performance related options
    pub rechunk: bool,
    pub n_threads: Option<usize>,
    pub low_memory: bool,
    // Row-wise options
    pub n_rows: Option<usize>,
    pub row_index: Option<RowIndex>,
    // Column-wise options
    pub columns: Option<Arc<[PlSmallStr]>>,
    pub projection: Option<Arc<Vec<usize>>>,
    pub schema: Option<SchemaRef>,
    pub schema_overwrite: Option<SchemaRef>,
    pub dtype_overwrite: Option<Arc<Vec<DataType>>>,
    // CSV-specific options
    pub parse_options: Arc<CsvParseOptions>,
    pub has_header: bool,
    pub chunk_size: usize,
    /// Skip rows according to the CSV spec.
    pub skip_rows: usize,
    /// Skip lines according to newline char (e.g. escaping will be ignored)
    pub skip_lines: usize,
    pub skip_rows_after_header: usize,
    pub infer_schema_length: Option<usize>,
    pub raise_if_empty: bool,
    pub ignore_errors: bool,
    pub fields_to_cast: Vec<Field>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct CsvParseOptions {
    pub separator: u8,
    pub quote_char: Option<u8>,
    pub eol_char: u8,
    pub encoding: CsvEncoding,
    pub null_values: Option<NullValues>,
    pub missing_is_null: bool,
    pub truncate_ragged_lines: bool,
    pub comment_prefix: Option<CommentPrefix>,
    pub try_parse_dates: bool,
    pub decimal_comma: bool,
}

impl Default for CsvReadOptions {
    fn default() -> Self {
        Self {
            path: None,

            rechunk: false,
            n_threads: None,
            low_memory: false,

            n_rows: None,
            row_index: None,

            columns: None,
            projection: None,
            schema: None,
            schema_overwrite: None,
            dtype_overwrite: None,

            parse_options: Default::default(),
            has_header: true,
            chunk_size: 1 << 18,
            skip_rows: 0,
            skip_lines: 0,
            skip_rows_after_header: 0,
            infer_schema_length: Some(100),
            raise_if_empty: true,
            ignore_errors: false,
            fields_to_cast: vec![],
        }
    }
}

/// Options related to parsing the CSV format.
impl Default for CsvParseOptions {
    fn default() -> Self {
        Self {
            separator: b',',
            quote_char: Some(b'"'),
            eol_char: b'\n',
            encoding: Default::default(),
            null_values: None,
            missing_is_null: true,
            truncate_ragged_lines: false,
            comment_prefix: None,
            try_parse_dates: false,
            decimal_comma: false,
        }
    }
}

impl CsvReadOptions {
    pub fn get_parse_options(&self) -> Arc<CsvParseOptions> {
        self.parse_options.clone()
    }

    pub fn with_path<P: Into<PathBuf>>(mut self, path: Option<P>) -> Self {
        self.path = path.map(|p| p.into());
        self
    }

    /// Whether to makes the columns contiguous in memory.
    pub fn with_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    /// Number of threads to use for reading. Defaults to the size of the polars
    /// thread pool.
    pub fn with_n_threads(mut self, n_threads: Option<usize>) -> Self {
        self.n_threads = n_threads;
        self
    }

    /// Reduce memory consumption at the expense of performance
    pub fn with_low_memory(mut self, low_memory: bool) -> Self {
        self.low_memory = low_memory;
        self
    }

    /// Limits the number of rows to read.
    pub fn with_n_rows(mut self, n_rows: Option<usize>) -> Self {
        self.n_rows = n_rows;
        self
    }

    /// Adds a row index column.
    pub fn with_row_index(mut self, row_index: Option<RowIndex>) -> Self {
        self.row_index = row_index;
        self
    }

    /// Which columns to select.
    pub fn with_columns(mut self, columns: Option<Arc<[PlSmallStr]>>) -> Self {
        self.columns = columns;
        self
    }

    /// Which columns to select denoted by their index. The index starts from 0
    /// (i.e. [0, 4] would select the 1st and 5th column).
    pub fn with_projection(mut self, projection: Option<Arc<Vec<usize>>>) -> Self {
        self.projection = projection;
        self
    }

    /// Set the schema to use for CSV file. The length of the schema must match
    /// the number of columns in the file. If this is [None], the schema is
    /// inferred from the file.
    pub fn with_schema(mut self, schema: Option<SchemaRef>) -> Self {
        self.schema = schema;
        self
    }

    /// Overwrites the data types in the schema by column name.
    pub fn with_schema_overwrite(mut self, schema_overwrite: Option<SchemaRef>) -> Self {
        self.schema_overwrite = schema_overwrite;
        self
    }

    /// Overwrite the dtypes in the schema in the order of the slice that's given.
    /// This is useful if you don't know the column names beforehand
    pub fn with_dtype_overwrite(mut self, dtype_overwrite: Option<Arc<Vec<DataType>>>) -> Self {
        self.dtype_overwrite = dtype_overwrite;
        self
    }

    /// Sets the CSV parsing options. See [map_parse_options][Self::map_parse_options]
    /// for an easier way to mutate them in-place.
    pub fn with_parse_options(mut self, parse_options: CsvParseOptions) -> Self {
        self.parse_options = Arc::new(parse_options);
        self
    }

    /// Sets whether the CSV file has a header row.
    pub fn with_has_header(mut self, has_header: bool) -> Self {
        self.has_header = has_header;
        self
    }

    /// Sets the chunk size used by the parser. This influences performance.
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Start reading after ``skip_rows`` rows. The header will be parsed at this
    /// offset. Note that we respect CSV escaping/comments when skipping rows.
    /// If you want to skip by newline char only, use `skip_lines`.
    pub fn with_skip_rows(mut self, skip_rows: usize) -> Self {
        self.skip_rows = skip_rows;
        self
    }

    /// Start reading after `skip_lines` lines. The header will be parsed at this
    /// offset. Note that CSV escaping will not be respected when skipping lines.
    /// If you want to skip valid CSV rows, use ``skip_rows``.
    pub fn with_skip_lines(mut self, skip_lines: usize) -> Self {
        self.skip_lines = skip_lines;
        self
    }

    /// Number of rows to skip after the header row.
    pub fn with_skip_rows_after_header(mut self, skip_rows_after_header: usize) -> Self {
        self.skip_rows_after_header = skip_rows_after_header;
        self
    }

    /// Set the number of rows to use when inferring the csv schema.
    /// The default is 100 rows.
    /// Setting to [None] will do a full table scan, which is very slow.
    pub fn with_infer_schema_length(mut self, infer_schema_length: Option<usize>) -> Self {
        self.infer_schema_length = infer_schema_length;
        self
    }

    /// Whether to raise an error if the frame is empty. By default an empty
    /// DataFrame is returned.
    pub fn with_raise_if_empty(mut self, raise_if_empty: bool) -> Self {
        self.raise_if_empty = raise_if_empty;
        self
    }

    /// Continue with next batch when a ParserError is encountered.
    pub fn with_ignore_errors(mut self, ignore_errors: bool) -> Self {
        self.ignore_errors = ignore_errors;
        self
    }

    /// Apply a function to the parse options.
    pub fn map_parse_options<F: Fn(CsvParseOptions) -> CsvParseOptions>(
        mut self,
        map_func: F,
    ) -> Self {
        let parse_options = Arc::unwrap_or_clone(self.parse_options);
        self.parse_options = Arc::new(map_func(parse_options));
        self
    }
}

impl CsvParseOptions {
    /// The character used to separate fields in the CSV file. This
    /// is most often a comma ','.
    pub fn with_separator(mut self, separator: u8) -> Self {
        self.separator = separator;
        self
    }

    /// Set the character used for field quoting. This is most often double
    /// quotes '"'. Set this to [None] to disable quote parsing.
    pub fn with_quote_char(mut self, quote_char: Option<u8>) -> Self {
        self.quote_char = quote_char;
        self
    }

    /// Set the character used to indicate an end-of-line (eol).
    pub fn with_eol_char(mut self, eol_char: u8) -> Self {
        self.eol_char = eol_char;
        self
    }

    /// Set the encoding used by the file.
    pub fn with_encoding(mut self, encoding: CsvEncoding) -> Self {
        self.encoding = encoding;
        self
    }

    /// Set values that will be interpreted as missing/null.
    ///
    /// Note: These values are matched before quote-parsing, so if the null values
    /// are quoted then those quotes also need to be included here.
    pub fn with_null_values(mut self, null_values: Option<NullValues>) -> Self {
        self.null_values = null_values;
        self
    }

    /// Treat missing fields as null.
    pub fn with_missing_is_null(mut self, missing_is_null: bool) -> Self {
        self.missing_is_null = missing_is_null;
        self
    }

    /// Truncate lines that are longer than the schema.
    pub fn with_truncate_ragged_lines(mut self, truncate_ragged_lines: bool) -> Self {
        self.truncate_ragged_lines = truncate_ragged_lines;
        self
    }

    /// Sets the comment prefix for this instance. Lines starting with this
    /// prefix will be ignored.
    pub fn with_comment_prefix<T: Into<CommentPrefix>>(
        mut self,
        comment_prefix: Option<T>,
    ) -> Self {
        self.comment_prefix = comment_prefix.map(Into::into);
        self
    }

    /// Automatically try to parse dates/datetimes and time. If parsing fails,
    /// columns remain of dtype [`DataType::String`].
    pub fn with_try_parse_dates(mut self, try_parse_dates: bool) -> Self {
        self.try_parse_dates = try_parse_dates;
        self
    }

    /// Parse floats with a comma as decimal separator.
    pub fn with_decimal_comma(mut self, decimal_comma: bool) -> Self {
        self.decimal_comma = decimal_comma;
        self
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum CsvEncoding {
    /// Utf8 encoding.
    #[default]
    Utf8,
    /// Utf8 encoding and unknown bytes are replaced with �.
    LossyUtf8,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum CommentPrefix {
    /// A single byte character that indicates the start of a comment line.
    Single(u8),
    /// A string that indicates the start of a comment line.
    /// This allows for multiple characters to be used as a comment identifier.
    Multi(PlSmallStr),
}

impl CommentPrefix {
    /// Creates a new `CommentPrefix` for the `Single` variant.
    pub fn new_single(prefix: u8) -> Self {
        CommentPrefix::Single(prefix)
    }

    /// Creates a new `CommentPrefix` for the `Multi` variant.
    pub fn new_multi(prefix: PlSmallStr) -> Self {
        CommentPrefix::Multi(prefix)
    }

    /// Creates a new `CommentPrefix` from a `&str`.
    pub fn new_from_str(prefix: &str) -> Self {
        if prefix.len() == 1 && prefix.chars().next().unwrap().is_ascii() {
            let c = prefix.as_bytes()[0];
            CommentPrefix::Single(c)
        } else {
            CommentPrefix::Multi(PlSmallStr::from_str(prefix))
        }
    }
}

impl From<&str> for CommentPrefix {
    fn from(value: &str) -> Self {
        Self::new_from_str(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum NullValues {
    /// A single value that's used for all columns
    AllColumnsSingle(PlSmallStr),
    /// Multiple values that are used for all columns
    AllColumns(Vec<PlSmallStr>),
    /// Tuples that map column names to null value of that column
    Named(Vec<(PlSmallStr, PlSmallStr)>),
}

impl NullValues {
    pub fn compile(self, schema: &Schema) -> PolarsResult<NullValuesCompiled> {
        Ok(match self {
            NullValues::AllColumnsSingle(v) => NullValuesCompiled::AllColumnsSingle(v),
            NullValues::AllColumns(v) => NullValuesCompiled::AllColumns(v),
            NullValues::Named(v) => {
                let mut null_values = vec![PlSmallStr::from_static(""); schema.len()];
                for (name, null_value) in v {
                    let i = schema.try_index_of(&name)?;
                    null_values[i] = null_value;
                }
                NullValuesCompiled::Columns(null_values)
            },
        })
    }
}

#[derive(Debug, Clone)]
pub enum NullValuesCompiled {
    /// A single value that's used for all columns
    AllColumnsSingle(PlSmallStr),
    // Multiple null values that are null for all columns
    AllColumns(Vec<PlSmallStr>),
    /// A different null value per column, computed from `NullValues::Named`
    Columns(Vec<PlSmallStr>),
}

impl NullValuesCompiled {
    /// # Safety
    ///
    /// The caller must ensure that `index` is in bounds
    pub(super) unsafe fn is_null(&self, field: &[u8], index: usize) -> bool {
        use NullValuesCompiled::*;
        match self {
            AllColumnsSingle(v) => v.as_bytes() == field,
            AllColumns(v) => v.iter().any(|v| v.as_bytes() == field),
            Columns(v) => {
                debug_assert!(index < v.len());
                v.get_unchecked(index).as_bytes() == field
            },
        }
    }
}
