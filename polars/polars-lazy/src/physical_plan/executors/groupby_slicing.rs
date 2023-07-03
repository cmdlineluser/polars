#[cfg(feature = "dynamic_groupby")]
use polars_core::frame::groupby::GroupBy;
#[cfg(feature = "dynamic_groupby")]
use polars_time::SlicingGroupOptions;

use super::*;

#[cfg_attr(not(feature = "dynamic_groupby"), allow(dead_code))]
pub(crate) struct GroupBySlicingExec {
    pub(crate) input: Box<dyn Executor>,
    //pub(crate) keys: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) aggs: Vec<Arc<dyn PhysicalExpr>>,
    #[cfg(feature = "dynamic_groupby")]
    pub(crate) options: SlicingGroupOptions,
    pub(crate) input_schema: SchemaRef,
    pub(crate) slice: Option<(i64, usize)>,
    pub(crate) apply: Option<Arc<dyn DataFrameUdf>>,
}

impl GroupBySlicingExec {
    #[cfg(feature = "dynamic_groupby")]
    fn execute_impl(
        &mut self,
        state: &mut ExecutionState,
        mut df: DataFrame,
    ) -> PolarsResult<DataFrame> {
        df.as_single_chunk_par();

        // Should this be done with POOL?
        let start = df.column(&self.options.start_column)?;
        let end = df.column(&self.options.end_column)?;

        let sa = start.u32()?;
        let sb = end.u32()?;

        let groups = sa
            .into_iter()
            .zip(sb.into_iter())
            .map(|(a, b)| {
                match (a, b) {
                    (Some(a), Some(b)) => {
                        let start = a as IdxSize;
                        let end = b as IdxSize;
                        let len = end - start as IdxSize;
                        [start, len]
                    }
                    _ => [0, 0], // Is this is wrong. What to do here?
                }
            })
            .collect();

        let groups = GroupsProxy::Slice {
            groups,
            rolling: false,
        };
        let gb = GroupBy::new(&df, vec![], groups, None);

        if let Some(f) = &self.apply {
            return gb.apply(move |df| f.call_udf(df));
        }

        let mut groups = gb.get_groups();

        #[allow(unused_assignments)]
        // it is unused because we only use it to keep the lifetime of sliced_group valid
        let mut sliced_groups = None;

        if let Some((offset, len)) = self.slice {
            sliced_groups = Some(groups.slice(offset, len));
            groups = sliced_groups.as_deref().unwrap();
        }

        state.expr_cache = Some(Default::default());
        let (mut columns, agg_columns) = POOL.install(|| {
            let get_columns = || gb.keys_sliced(self.slice);

            let get_agg = || {
                self.aggs
                    .par_iter()
                    .map(|expr| {
                        let agg = expr.evaluate_on_groups(&df, groups, state)?.finalize();
                        polars_ensure!(
                            agg.len() == groups.len(),
                            agg_len = agg.len(),
                            groups.len()
                        );
                        Ok(agg)
                    })
                    .collect::<PolarsResult<Vec<_>>>()
            };

            rayon::join(get_columns, get_agg)
        });
        let agg_columns = agg_columns?;
        state.expr_cache = None;

        columns.extend_from_slice(&agg_columns);
        DataFrame::new(columns)
    }
}

impl Executor for GroupBySlicingExec {
    #[cfg(not(feature = "dynamic_groupby"))]
    fn execute(&mut self, _state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        panic!("activate feature dynamic_groupby")
    }

    #[cfg(feature = "dynamic_groupby")]
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                println!("run GroupbySlicingExec")
            }
        }
        let df = self.input.execute(state)?;
        let profile_name = if state.has_node_timer() {
            Cow::Borrowed("groupby_slicing")
        } else {
            Cow::Borrowed("")
        };

        if state.has_node_timer() {
            let new_state = state.clone();
            new_state.record(|| self.execute_impl(state, df), profile_name)
        } else {
            self.execute_impl(state, df)
        }
    }
}
