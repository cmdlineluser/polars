//! Implementations of the ChunkCast Trait.

use std::borrow::Cow;

use polars_compute::cast::CastOptionsImpl;
#[cfg(feature = "serde-lazy")]
use serde::{Deserialize, Serialize};

use super::flags::StatisticsFlags;
#[cfg(feature = "dtype-datetime")]
use crate::prelude::DataType::Datetime;
use crate::prelude::*;
use crate::utils::handle_casting_failures;

#[derive(Copy, Clone, Debug, Default, PartialEq, Hash, Eq)]
#[cfg_attr(feature = "serde-lazy", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[repr(u8)]
pub enum CastOptions {
    /// Raises on overflow
    #[default]
    Strict,
    /// Overflow is replaced with null
    NonStrict,
    /// Allows wrapping overflow
    Overflowing,
}

impl CastOptions {
    pub fn is_strict(&self) -> bool {
        matches!(self, CastOptions::Strict)
    }
}

impl From<CastOptions> for CastOptionsImpl {
    fn from(value: CastOptions) -> Self {
        let wrapped = match value {
            CastOptions::Strict | CastOptions::NonStrict => false,
            CastOptions::Overflowing => true,
        };
        CastOptionsImpl {
            wrapped,
            partial: false,
        }
    }
}

pub(crate) fn cast_chunks(
    chunks: &[ArrayRef],
    dtype: &DataType,
    options: CastOptions,
) -> PolarsResult<Vec<ArrayRef>> {
    let check_nulls = matches!(options, CastOptions::Strict);
    let options = options.into();

    let arrow_dtype = dtype.try_to_arrow(CompatLevel::newest())?;
    chunks
        .iter()
        .map(|arr| {
            let out = polars_compute::cast::cast(arr.as_ref(), &arrow_dtype, options);
            if check_nulls {
                out.and_then(|new| {
                    polars_ensure!(arr.null_count() == new.null_count(), ComputeError: "strict cast failed");
                    Ok(new)
                })

            } else {
                out
            }
        })
        .collect::<PolarsResult<Vec<_>>>()
}

fn cast_impl_inner(
    name: PlSmallStr,
    chunks: &[ArrayRef],
    dtype: &DataType,
    options: CastOptions,
) -> PolarsResult<Series> {
    let chunks = match dtype {
        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(_, _) => {
            let mut chunks = cast_chunks(chunks, dtype, options)?;
            // @NOTE: We cannot cast here as that will lower the scale.
            for chunk in chunks.iter_mut() {
                *chunk = std::mem::take(
                    chunk
                        .as_any_mut()
                        .downcast_mut::<PrimitiveArray<i128>>()
                        .unwrap(),
                )
                .to(ArrowDataType::Int128)
                .to_boxed();
            }
            chunks
        },
        _ => cast_chunks(chunks, &dtype.to_physical(), options)?,
    };

    let out = Series::try_from((name, chunks))?;
    use DataType::*;
    let out = match dtype {
        Date => out.into_date(),
        Datetime(tu, tz) => match tz {
            #[cfg(feature = "timezones")]
            Some(tz) => {
                TimeZone::validate_time_zone(tz)?;
                out.into_datetime(*tu, Some(tz.clone()))
            },
            _ => out.into_datetime(*tu, None),
        },
        Duration(tu) => out.into_duration(*tu),
        #[cfg(feature = "dtype-time")]
        Time => out.into_time(),
        #[cfg(feature = "dtype-decimal")]
        Decimal(precision, scale) => out.into_decimal(*precision, scale.unwrap_or(0))?,
        _ => out,
    };

    Ok(out)
}

fn cast_impl(
    name: PlSmallStr,
    chunks: &[ArrayRef],
    dtype: &DataType,
    options: CastOptions,
) -> PolarsResult<Series> {
    cast_impl_inner(name, chunks, dtype, options)
}

#[cfg(feature = "dtype-struct")]
fn cast_single_to_struct(
    name: PlSmallStr,
    chunks: &[ArrayRef],
    fields: &[Field],
    options: CastOptions,
) -> PolarsResult<Series> {
    polars_ensure!(fields.len() == 1, InvalidOperation: "must specify one field in the struct");
    let mut new_fields = Vec::with_capacity(fields.len());
    // cast to first field dtype
    let mut fields = fields.iter();
    let fld = fields.next().unwrap();
    let s = cast_impl_inner(fld.name.clone(), chunks, &fld.dtype, options)?;
    let length = s.len();
    new_fields.push(s);

    for fld in fields {
        new_fields.push(Series::full_null(fld.name.clone(), length, &fld.dtype));
    }

    StructChunked::from_series(name, length, new_fields.iter()).map(|ca| ca.into_series())
}

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn cast_impl(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Series> {
        if self.dtype() == dtype {
            // SAFETY: chunks are correct dtype
            let mut out = unsafe {
                Series::from_chunks_and_dtype_unchecked(
                    self.name().clone(),
                    self.chunks.clone(),
                    dtype,
                )
            };
            out.set_sorted_flag(self.is_sorted_flag());
            return Ok(out);
        }
        match dtype {
            // LEGACY
            // TODO @ cat-rework: remove after exposing to/from physical functions.
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(cats, _mapping) => {
                let s = self.cast_with_options(&cats.physical().dtype(), options)?;
                with_match_categorical_physical_type!(cats.physical(), |$C| {
                    // SAFETY: we are guarded by the type system.
                    type PhysCa = ChunkedArray<<$C as PolarsCategoricalType>::PolarsPhysical>;
                    let ca: &PhysCa = s.as_ref().as_ref();
                    Ok(CategoricalChunked::<$C>::from_cats_and_dtype(ca.clone(), dtype.clone())
                        .into_series())
                })
            },

            // LEGACY
            // TODO @ cat-rework: remove after exposing to/from physical functions.
            #[cfg(feature = "dtype-categorical")]
            DataType::Enum(fcats, _mapping) => {
                let s = self.cast_with_options(&fcats.physical().dtype(), options)?;
                with_match_categorical_physical_type!(fcats.physical(), |$C| {
                    // SAFETY: we are guarded by the type system.
                    type PhysCa = ChunkedArray<<$C as PolarsCategoricalType>::PolarsPhysical>;
                    let ca: &PhysCa = s.as_ref().as_ref();
                    Ok(CategoricalChunked::<$C>::from_cats_and_dtype(ca.clone(), dtype.clone()).into_series())
                })
            },

            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => {
                cast_single_to_struct(self.name().clone(), &self.chunks, fields, options)
            },
            _ => cast_impl_inner(self.name().clone(), &self.chunks, dtype, options).map(|mut s| {
                // maintain sorted if data types
                // - remain signed
                // - unsigned -> signed
                // this may still fail with overflow?
                let to_signed = dtype.is_signed_integer();
                let unsigned2unsigned =
                    self.dtype().is_unsigned_integer() && dtype.is_unsigned_integer();
                let allowed = to_signed || unsigned2unsigned;

                if (allowed)
                    && (s.null_count() == self.null_count())
                    // physical to logicals
                    || (self.dtype().to_physical() == dtype.to_physical())
                {
                    let is_sorted = self.is_sorted_flag();
                    s.set_sorted_flag(is_sorted)
                }
                s
            }),
        }
    }
}

impl<T> ChunkCast for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn cast_with_options(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Series> {
        self.cast_impl(dtype, options)
    }

    unsafe fn cast_unchecked(&self, dtype: &DataType) -> PolarsResult<Series> {
        match dtype {
            // LEGACY
            // TODO @ cat-rework: remove after exposing to/from physical functions.
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(cats, _mapping) => {
                polars_ensure!(self.dtype() == &cats.physical().dtype(), ComputeError: "cannot cast numeric types to 'Categorical'");
                with_match_categorical_physical_type!(cats.physical(), |$C| {
                    // SAFETY: we are guarded by the type system.
                    type PhysCa = ChunkedArray<<$C as PolarsCategoricalType>::PolarsPhysical>;
                    let ca = unsafe { &*(self as *const ChunkedArray<T> as *const PhysCa) };
                    Ok(CategoricalChunked::<$C>::from_cats_and_dtype_unchecked(ca.clone(), dtype.clone())
                        .into_series())
                })
            },

            // LEGACY
            // TODO @ cat-rework: remove after exposing to/from physical functions.
            #[cfg(feature = "dtype-categorical")]
            DataType::Enum(fcats, _mapping) => {
                polars_ensure!(self.dtype() == &fcats.physical().dtype(), ComputeError: "cannot cast numeric types to 'Enum'");
                with_match_categorical_physical_type!(fcats.physical(), |$C| {
                    // SAFETY: we are guarded by the type system.
                    type PhysCa = ChunkedArray<<$C as PolarsCategoricalType>::PolarsPhysical>;
                    let ca = unsafe { &*(self as *const ChunkedArray<T> as *const PhysCa) };
                    Ok(CategoricalChunked::<$C>::from_cats_and_dtype_unchecked(ca.clone(), dtype.clone()).into_series())
                })
            },

            _ => self.cast_impl(dtype, CastOptions::Overflowing),
        }
    }
}

impl ChunkCast for StringChunked {
    fn cast_with_options(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Series> {
        match dtype {
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(cats, _mapping) => {
                with_match_categorical_physical_type!(cats.physical(), |$C| {
                    Ok(CategoricalChunked::<$C>::from_str_iter(self.name().clone(), dtype.clone(), self.iter())?
                        .into_series())
                })
            },
            #[cfg(feature = "dtype-categorical")]
            DataType::Enum(fcats, _mapping) => {
                let ret = with_match_categorical_physical_type!(fcats.physical(), |$C| {
                    CategoricalChunked::<$C>::from_str_iter(self.name().clone(), dtype.clone(), self.iter())?
                        .into_series()
                });

                if options.is_strict() && self.null_count() != ret.null_count() {
                    handle_casting_failures(&self.clone().into_series(), &ret)?;
                }

                Ok(ret)
            },
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => {
                cast_single_to_struct(self.name().clone(), &self.chunks, fields, options)
            },
            #[cfg(feature = "dtype-decimal")]
            DataType::Decimal(precision, scale) => match (precision, scale) {
                (precision, Some(scale)) => {
                    let chunks = self.downcast_iter().map(|arr| {
                        polars_compute::cast::binview_to_decimal(
                            &arr.to_binview(),
                            *precision,
                            *scale,
                        )
                        .to(ArrowDataType::Int128)
                    });
                    Ok(Int128Chunked::from_chunk_iter(self.name().clone(), chunks)
                        .into_decimal_unchecked(*precision, *scale)
                        .into_series())
                },
                (None, None) => self.to_decimal(100),
                _ => {
                    polars_bail!(ComputeError: "expected 'precision' or 'scale' when casting to Decimal")
                },
            },
            #[cfg(feature = "dtype-date")]
            DataType::Date => {
                let result = cast_chunks(&self.chunks, dtype, options)?;
                let out = Series::try_from((self.name().clone(), result))?;
                Ok(out)
            },
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(time_unit, time_zone) => match time_zone {
                #[cfg(feature = "timezones")]
                Some(time_zone) => {
                    TimeZone::validate_time_zone(time_zone)?;
                    let result = cast_chunks(
                        &self.chunks,
                        &Datetime(time_unit.to_owned(), Some(time_zone.clone())),
                        options,
                    )?;
                    Series::try_from((self.name().clone(), result))
                },
                _ => {
                    let result =
                        cast_chunks(&self.chunks, &Datetime(time_unit.to_owned(), None), options)?;
                    Series::try_from((self.name().clone(), result))
                },
            },
            _ => cast_impl(self.name().clone(), &self.chunks, dtype, options),
        }
    }

    unsafe fn cast_unchecked(&self, dtype: &DataType) -> PolarsResult<Series> {
        self.cast_with_options(dtype, CastOptions::Overflowing)
    }
}

impl BinaryChunked {
    /// # Safety
    /// String is not validated
    pub unsafe fn to_string_unchecked(&self) -> StringChunked {
        let chunks = self
            .downcast_iter()
            .map(|arr| unsafe { arr.to_utf8view_unchecked() }.boxed())
            .collect();
        let field = Arc::new(Field::new(self.name().clone(), DataType::String));

        let mut ca = StringChunked::new_with_compute_len(field, chunks);

        use StatisticsFlags as F;
        ca.retain_flags_from(self, F::IS_SORTED_ANY | F::CAN_FAST_EXPLODE_LIST);
        ca
    }
}

impl StringChunked {
    pub fn as_binary(&self) -> BinaryChunked {
        let chunks = self
            .downcast_iter()
            .map(|arr| arr.to_binview().boxed())
            .collect();
        let field = Arc::new(Field::new(self.name().clone(), DataType::Binary));

        let mut ca = BinaryChunked::new_with_compute_len(field, chunks);

        use StatisticsFlags as F;
        ca.retain_flags_from(self, F::IS_SORTED_ANY | F::CAN_FAST_EXPLODE_LIST);
        ca
    }
}

impl ChunkCast for BinaryChunked {
    fn cast_with_options(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Series> {
        match dtype {
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => {
                cast_single_to_struct(self.name().clone(), &self.chunks, fields, options)
            },
            _ => cast_impl(self.name().clone(), &self.chunks, dtype, options),
        }
    }

    unsafe fn cast_unchecked(&self, dtype: &DataType) -> PolarsResult<Series> {
        match dtype {
            DataType::String => unsafe { Ok(self.to_string_unchecked().into_series()) },
            _ => self.cast_with_options(dtype, CastOptions::Overflowing),
        }
    }
}

impl ChunkCast for BinaryOffsetChunked {
    fn cast_with_options(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Series> {
        match dtype {
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => {
                cast_single_to_struct(self.name().clone(), &self.chunks, fields, options)
            },
            _ => cast_impl(self.name().clone(), &self.chunks, dtype, options),
        }
    }

    unsafe fn cast_unchecked(&self, dtype: &DataType) -> PolarsResult<Series> {
        self.cast_with_options(dtype, CastOptions::Overflowing)
    }
}

impl ChunkCast for BooleanChunked {
    fn cast_with_options(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Series> {
        match dtype {
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => {
                cast_single_to_struct(self.name().clone(), &self.chunks, fields, options)
            },
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, _) | DataType::Enum(_, _) => {
                polars_bail!(InvalidOperation: "cannot cast Boolean to Categorical");
            },
            _ => cast_impl(self.name().clone(), &self.chunks, dtype, options),
        }
    }

    unsafe fn cast_unchecked(&self, dtype: &DataType) -> PolarsResult<Series> {
        self.cast_with_options(dtype, CastOptions::Overflowing)
    }
}

/// We cannot cast anything to or from List/LargeList
/// So this implementation casts the inner type
impl ChunkCast for ListChunked {
    fn cast_with_options(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Series> {
        let ca = self
            .trim_lists_to_normalized_offsets()
            .map_or(Cow::Borrowed(self), Cow::Owned);
        let ca = ca.propagate_nulls().map_or(ca, Cow::Owned);

        use DataType::*;
        match dtype {
            List(child_type) => {
                match (ca.inner_dtype(), &**child_type) {
                    (old, new) if old == new => Ok(ca.into_owned().into_series()),
                    // TODO @ cat-rework: can we implement this now?
                    #[cfg(feature = "dtype-categorical")]
                    (dt, Categorical(_, _) | Enum(_, _))
                        if !matches!(dt, Categorical(_, _) | Enum(_, _) | String | Null) =>
                    {
                        polars_bail!(InvalidOperation: "cannot cast List inner type: '{:?}' to Categorical", dt)
                    },
                    _ => {
                        // ensure the inner logical type bubbles up
                        let (arr, child_type) = cast_list(ca.as_ref(), child_type, options)?;
                        // SAFETY: we just cast so the dtype matches.
                        // we must take this path to correct for physical types.
                        unsafe {
                            Ok(Series::from_chunks_and_dtype_unchecked(
                                ca.name().clone(),
                                vec![arr],
                                &List(Box::new(child_type)),
                            ))
                        }
                    },
                }
            },
            #[cfg(feature = "dtype-array")]
            Array(child_type, width) => {
                let physical_type = dtype.to_physical();

                // TODO @ cat-rework: can we implement this now?
                // TODO!: properly implement this recursively.
                #[cfg(feature = "dtype-categorical")]
                polars_ensure!(!matches!(&**child_type, Categorical(_, _)), InvalidOperation: "array of categorical is not yet supported");

                // cast to the physical type to avoid logical chunks.
                let chunks = cast_chunks(ca.chunks(), &physical_type, options)?;
                // SAFETY: we just cast so the dtype matches.
                // we must take this path to correct for physical types.
                unsafe {
                    Ok(Series::from_chunks_and_dtype_unchecked(
                        ca.name().clone(),
                        chunks,
                        &Array(child_type.clone(), *width),
                    ))
                }
            },
            #[cfg(feature = "dtype-u8")]
            Binary => {
                polars_ensure!(
                    matches!(self.inner_dtype(), UInt8),
                    InvalidOperation: "cannot cast List type (inner: '{:?}', to: '{:?}')",
                    self.inner_dtype(),
                    dtype,
                );
                let chunks = cast_chunks(self.chunks(), &DataType::Binary, options)?;

                // SAFETY: we just cast so the dtype matches.
                unsafe {
                    Ok(Series::from_chunks_and_dtype_unchecked(
                        self.name().clone(),
                        chunks,
                        &DataType::Binary,
                    ))
                }
            },
            _ => {
                polars_bail!(
                    InvalidOperation: "cannot cast List type (inner: '{:?}', to: '{:?}')",
                    ca.inner_dtype(),
                    dtype,
                )
            },
        }
    }

    unsafe fn cast_unchecked(&self, dtype: &DataType) -> PolarsResult<Series> {
        use DataType::*;
        match dtype {
            List(child_type) => cast_list_unchecked(self, child_type),
            _ => self.cast_with_options(dtype, CastOptions::Overflowing),
        }
    }
}

/// We cannot cast anything to or from List/LargeList
/// So this implementation casts the inner type
#[cfg(feature = "dtype-array")]
impl ChunkCast for ArrayChunked {
    fn cast_with_options(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Series> {
        let ca = self
            .trim_lists_to_normalized_offsets()
            .map_or(Cow::Borrowed(self), Cow::Owned);
        let ca = ca.propagate_nulls().map_or(ca, Cow::Owned);

        use DataType::*;
        match dtype {
            Array(child_type, width) => {
                polars_ensure!(
                    *width == ca.width(),
                    InvalidOperation: "cannot cast Array to a different width"
                );

                match (ca.inner_dtype(), &**child_type) {
                    (old, new) if old == new => Ok(ca.into_owned().into_series()),
                    // TODO @ cat-rework: can we implement this now?
                    #[cfg(feature = "dtype-categorical")]
                    (dt, Categorical(_, _) | Enum(_, _)) if !matches!(dt, String) => {
                        polars_bail!(InvalidOperation: "cannot cast Array inner type: '{:?}' to dtype: {:?}", dt, child_type)
                    },
                    _ => {
                        // ensure the inner logical type bubbles up
                        let (arr, child_type) =
                            cast_fixed_size_list(ca.as_ref(), child_type, options)?;
                        // SAFETY: we just cast so the dtype matches.
                        // we must take this path to correct for physical types.
                        unsafe {
                            Ok(Series::from_chunks_and_dtype_unchecked(
                                ca.name().clone(),
                                vec![arr],
                                &Array(Box::new(child_type), *width),
                            ))
                        }
                    },
                }
            },
            List(child_type) => {
                let physical_type = dtype.to_physical();
                // cast to the physical type to avoid logical chunks.
                let chunks = cast_chunks(ca.chunks(), &physical_type, options)?;
                // SAFETY: we just cast so the dtype matches.
                // we must take this path to correct for physical types.
                unsafe {
                    Ok(Series::from_chunks_and_dtype_unchecked(
                        ca.name().clone(),
                        chunks,
                        &List(child_type.clone()),
                    ))
                }
            },
            _ => {
                polars_bail!(
                    InvalidOperation: "cannot cast Array type (inner: '{:?}', to: '{:?}')",
                    ca.inner_dtype(),
                    dtype,
                )
            },
        }
    }

    unsafe fn cast_unchecked(&self, dtype: &DataType) -> PolarsResult<Series> {
        self.cast_with_options(dtype, CastOptions::Overflowing)
    }
}

// Returns inner data type. This is needed because a cast can instantiate the dtype inner
// values for instance with categoricals
fn cast_list(
    ca: &ListChunked,
    child_type: &DataType,
    options: CastOptions,
) -> PolarsResult<(ArrayRef, DataType)> {
    // We still rechunk because we must bubble up a single data-type
    // TODO!: consider a version that works on chunks and merges the data-types and arrays.
    let ca = ca.rechunk();
    let arr = ca.downcast_as_array();
    // SAFETY: inner dtype is passed correctly
    let s = unsafe {
        Series::from_chunks_and_dtype_unchecked(
            PlSmallStr::EMPTY,
            vec![arr.values().clone()],
            ca.inner_dtype(),
        )
    };
    let new_inner = s.cast_with_options(child_type, options)?;

    let inner_dtype = new_inner.dtype().clone();
    debug_assert_eq!(&inner_dtype, child_type);

    let new_values = new_inner.array_ref(0).clone();

    let dtype = ListArray::<i64>::default_datatype(new_values.dtype().clone());
    let new_arr = ListArray::<i64>::new(
        dtype,
        arr.offsets().clone(),
        new_values,
        arr.validity().cloned(),
    );
    Ok((new_arr.boxed(), inner_dtype))
}

unsafe fn cast_list_unchecked(ca: &ListChunked, child_type: &DataType) -> PolarsResult<Series> {
    // TODO! add chunked, but this must correct for list offsets.
    let ca = ca.rechunk();
    let arr = ca.downcast_as_array();
    // SAFETY: inner dtype is passed correctly
    let s = unsafe {
        Series::from_chunks_and_dtype_unchecked(
            PlSmallStr::EMPTY,
            vec![arr.values().clone()],
            ca.inner_dtype(),
        )
    };
    let new_inner = s.cast_unchecked(child_type)?;
    let new_values = new_inner.array_ref(0).clone();

    let dtype = ListArray::<i64>::default_datatype(new_values.dtype().clone());
    let new_arr = ListArray::<i64>::new(
        dtype,
        arr.offsets().clone(),
        new_values,
        arr.validity().cloned(),
    );
    Ok(ListChunked::from_chunks_and_dtype_unchecked(
        ca.name().clone(),
        vec![Box::new(new_arr)],
        DataType::List(Box::new(child_type.clone())),
    )
    .into_series())
}

// Returns inner data type. This is needed because a cast can instantiate the dtype inner
// values for instance with categoricals
#[cfg(feature = "dtype-array")]
fn cast_fixed_size_list(
    ca: &ArrayChunked,
    child_type: &DataType,
    options: CastOptions,
) -> PolarsResult<(ArrayRef, DataType)> {
    let ca = ca.rechunk();
    let arr = ca.downcast_as_array();
    // SAFETY: inner dtype is passed correctly
    let s = unsafe {
        Series::from_chunks_and_dtype_unchecked(
            PlSmallStr::EMPTY,
            vec![arr.values().clone()],
            ca.inner_dtype(),
        )
    };
    let new_inner = s.cast_with_options(child_type, options)?;

    let inner_dtype = new_inner.dtype().clone();
    debug_assert_eq!(&inner_dtype, child_type);

    let new_values = new_inner.array_ref(0).clone();

    let dtype = FixedSizeListArray::default_datatype(new_values.dtype().clone(), ca.width());
    let new_arr = FixedSizeListArray::new(dtype, ca.len(), new_values, arr.validity().cloned());
    Ok((Box::new(new_arr), inner_dtype))
}

#[cfg(test)]
mod test {
    use crate::chunked_array::cast::CastOptions;
    use crate::prelude::*;

    #[test]
    fn test_cast_list() -> PolarsResult<()> {
        let mut builder = ListPrimitiveChunkedBuilder::<Int32Type>::new(
            PlSmallStr::from_static("a"),
            10,
            10,
            DataType::Int32,
        );
        builder.append_opt_slice(Some(&[1i32, 2, 3]));
        builder.append_opt_slice(Some(&[1i32, 2, 3]));
        let ca = builder.finish();

        let new = ca.cast_with_options(
            &DataType::List(DataType::Float64.into()),
            CastOptions::Strict,
        )?;

        assert_eq!(new.dtype(), &DataType::List(DataType::Float64.into()));
        Ok(())
    }

    #[test]
    #[cfg(feature = "dtype-categorical")]
    fn test_cast_noop() {
        // check if we can cast categorical twice without panic
        let ca = StringChunked::new(PlSmallStr::from_static("foo"), &["bar", "ham"]);
        let cats = Categories::global();
        let out = ca
            .cast_with_options(
                &DataType::from_categories(cats.clone()),
                CastOptions::Strict,
            )
            .unwrap();
        let out = out.cast(&DataType::from_categories(cats)).unwrap();
        assert!(matches!(out.dtype(), &DataType::Categorical(_, _)))
    }
}
