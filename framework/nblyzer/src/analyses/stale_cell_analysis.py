# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from copy import deepcopy
from ..IR.intermediate_representations import IntermediateRepresentations
from .runner.runners import Runner
from .runner.analyses_utils import find_changed_vars, AssignParserVisitor, get_all_unbound_vars
from .analysis import Analysis
from .abs_states.code_impact_abs_state import CodeImpactAS
from .runner.analysis_results import Result
from .runner.stats import Stats
from simple_cfg.cfg_nodes import Node
import ast

class StaleCellAnalysis(Analysis):
    def __init__(self) -> None:
        super().__init__()
        self.k = 2
        self.abstract_state = CodeImpactAS()
        self.stats = []

    def F_transformer(self, cfg_node: Node, a_state: CodeImpactAS, cell_IR: IntermediateRepresentations):
        as_transformed: CodeImpactAS = deepcopy(a_state)
        if not cfg_node.ast_node:
            return a_state
        if isinstance(cfg_node.ast_node, ast.Assign):
            assign_parser = AssignParserVisitor()
            assign_parser.parse_assign(cfg_node.ast_node)
            def_vars = assign_parser.def_variables
            assigned_vars = assign_parser.assigned_variables
            for def_var in def_vars:
                as_transformed.set_var_level(def_var, -1000)
                for var in assigned_vars & as_transformed.impacted_variables.keys() - def_vars - self.imports:
                    if as_transformed.impacted_variables[var] != -1:
                        if var in cell_IR.UDA.def_use_chains.unbound_names and var not in cell_IR.UDA.defined_vars:
                            as_transformed.set_var_level(def_var, as_transformed.impacted_variables[var] + 1)
                        else:
                            as_transformed.set_var_level(def_var, as_transformed.impacted_variables[var])

        if isinstance(cfg_node.ast_node, ast.Name):
            if cfg_node.label in as_transformed.impacted_variables.keys() and as_transformed.impacted_variables[cfg_node.label] != -1:
                if cfg_node.label in cell_IR.UDA.def_use_chains.unbound_names:
                    as_transformed.set_var_level(cfg_node.label + "_usage", as_transformed.impacted_variables[cfg_node.label] + 1)
                else:
                    as_transformed.set_var_level(cfg_node.label + "_usage", as_transformed.impacted_variables[cfg_node.label])
        return as_transformed

    def combine_states(self, states: list[CodeImpactAS]):
        res_state = CodeImpactAS(K = self.k)
        for s in states:
            res_state.aug_join(s)
        return res_state

    def summarize_result(self, result: Result) -> Result:
        summarized_result: Result = result.distinct_errors()
        return summarized_result

    def _prepare_init_as(self, new_cell_IR, old_cell_IR):
        init_as: CodeImpactAS = deepcopy(self.abstract_state)
        changed_vars: set[str] = find_changed_vars(new_cell_IR, old_cell_IR)
        for var in changed_vars:
            init_as.impacted_variables[var] = 0
        for var in new_cell_IR.UDA.defined_vars.keys():
            if (var in init_as.impacted_variables and init_as.impacted_variables[var] > 0):
                init_as.impacted_variables[var] = 0
        return init_as

    def _run_fixpoint_analysis(self, notebook_IR, old_cell_IR=None, level=20, filename=""):
        self._find_all_imports(notebook_IR)
        self.all_unbound_vars = get_all_unbound_vars(notebook_IR)
        init_as: CodeImpactAS = self._prepare_init_as(
            notebook_IR[old_cell_IR.cell_id], old_cell_IR
        )
        stat = Stats(old_cell_IR.cell_id, filename)
        stat.log_start()
        runner: Runner = Runner(stat, defaultdict(CodeImpactAS), notebook_IR)
        result: Result = runner.inter_fixpoint_runner(
            self,
            old_cell_IR.cell_id,
            abstract_state=init_as,
            K=level,
            cpath=[],
            results=Result()
        )
        stat.log_end()
        self.stats.append(stat)
        return runner, result, init_as

    def _find_all_imports(self, notebook_IR: dict[str, IntermediateRepresentations]):
        self.imports = set()
        for cell_IR in notebook_IR.values():
            self.imports.update(cell_IR.UDA.imports)

    def analyze_notebook(self, notebook_IR, old_cell_IR=None, level=20, filename=""):
        self._find_all_imports(notebook_IR)
        if bool(self.calculate_pre(notebook_IR[old_cell_IR.cell_id])):
            return Result()

        return self._run_fixpoint_analysis(notebook_IR, old_cell_IR, level, filename)[1]

    def update_abstract_state(self, cell_IR, notebook_IR):
        self.find_necessary_cells(notebook_IR)
        old_cell_IR: IntermediateRepresentations = IntermediateRepresentations(
            cell_IR.last_ran_code, cell_IR.cell_id
        )
        runner, _, init_as = self._run_fixpoint_analysis(notebook_IR, old_cell_IR)
        for abs in runner.error_states:
            for var, level in abs.impacted_variables.items():
                if level == 1:
                    self.abstract_state.impacted_variables[var] = level
        for var, level in init_as.impacted_variables.items():
            if level == 0:
                self.abstract_state.impacted_variables[var] = -1

    def filter_state(self, abstract_state: CodeImpactAS, target_val: int) -> set[str]:
        return {key for key,val in abstract_state.impacted_variables.items() if val == target_val}

    def phi_condition(
        self,
        source_AS: CodeImpactAS,
        pre_summary: set,
        source_IR: IntermediateRepresentations,
        target_IR: IntermediateRepresentations,
        K: int,
    ):
        if not bool(self.all_unbound_vars & source_IR.UDA.defined_vars.keys()):
            return False
        
        if K == 2 and source_AS.max_domain_value <= 1:
            filtered_proj = self.filter_state(source_AS, 1)
            if not bool(filtered_proj & pre_summary):
                return False

        if K == 3 and source_AS.max_domain_value <= 0:
                filtered_proj = self.filter_state(source_AS, 0)
                if not bool(filtered_proj & pre_summary):
                    return False

        if not bool(target_IR.UDA.defined_vars) or not bool(self.all_unbound_vars & target_IR.UDA.defined_vars.keys()):
            filtered_proj = self.filter_state(source_AS, 1)
            if not bool(filtered_proj & pre_summary):
                return False

        return pre_summary <= source_AS.projection()

    def calculate_pre(self, cell_IR: IntermediateRepresentations):
        return cell_IR.UDA.unbound_final - self.imports

    def trivial_transformation(self, cell_IR: IntermediateRepresentations, abstract_state: CodeImpactAS):
        active_state = {key for key,val in abstract_state.impacted_variables.items() if val >= 0}
        return not bool(cell_IR.UDA.unbound_final & active_state)
