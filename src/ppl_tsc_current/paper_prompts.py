"""Figure A.2 fixed wording, with runtime information only in placeholders.

PDF line wrapping is removed; apostrophes are ASCII. The figure's spelling
`Constrains` and `satisfy on` are retained. No scoring or verifier changes.
"""
import re

TEMPLATE_ID = 'paper_figure_a2_v1'

TEMPLATES={
 'initialization':'''Task: Discover a student utility function that approximates the teacher model's decision behavior. The function Compute_utility(state, phase_id) assigns a utility score to each candidate phase; the phase with the highest score is activated. Prioritize diverse and expressive function structures during initialization.

Context: The available input features are {features}.

Constrains: The implementation must satisfy on {syntactic}, {mathematical}, and {semantic}.

Output: Provide the complete Python function Compute_utility(state, phase_id) that returns a float utility value, followed by a brief explanation (2–3 sentences) describing the design rationale.''',
 'evolution':'''Task: Evolve an improved offspring utility function from the given parent. The function Compute_utility(state, phase_id) assigns a utility score to each candidate phase; the phase with the highest score is activated. The offspring must be structurally distinct from the parent and combine raw features meaningfully to avoid trivial heuristics.

Context: The parent function is {parent}. Evaluation information is provided in {evaluation}. The available input features are {features}.

Requirements: Same as in the initialization prompt.

Output: Same as in the initialization prompt.''',
 'repair':'''Task: Repair an invalid utility function that violates specified constraints. The function Compute_utility(state, phase_id) assigns a utility score to each candidate phase; the phase with the highest score is activated. The repaired function must preserve the original behavioral intent as much as possible while strictly satisfying all constraints and eliminating any invalid operations.

Context: The function to be repaired is {parent}. The identified constraint violations are detailed in {violations}. The available input features are {features}.

Requirements: Same as in the initialization prompt.

Output: Same as in the initialization prompt.'''
}

BINDING='''EXECUTABLE INTERFACE BINDING:
The manuscript symbol Compute_utility(state, phase_id) denotes this runtime interface:
def compute_utility(state, phase_id=None, feature_stats=None):
Emit this lowercase name and these arguments. feature_stats contains only available frozen source measurement statistics; phase_active is a declared binary-support convention (0/1, mean 0.5, std 0.5), not measured phase frequencies. Unsupported teacher-only fields and their statistics are unavailable. The traffic features and decision mode below define exactly what a candidate phase/action means, including binary KEEP/SWITCH when applicable.
RESPONSE SERIALIZATION:
Provide one Python code block containing ### Function, the complete function (with its NumPy import), and ### Explanation followed by a brief rationale. This supplies the executable function and explanation requested by the template.'''

def hard_requirements(prompt):
 match=re.search(r'HARD REQUIREMENTS:\s*(.*?)(?:OUTPUT FORMAT \(STRICT\):|Output format \(strict\):)',prompt,re.S)
 if not match:raise ValueError('Missing baseline requirements; refuse to lose them')
 return match.group(1).strip()

def render(generator,stage,baseline_prompt,parent=None,explanation=None,feedback=None):
 # The exact source/feature contract is shared with the baseline. All added
 # implementation information stays inside the figure's feature placeholder.
 binary=generator._decision_mode()=='binary_keep_switch'
 contract=generator._build_binary_data_contract_text() if binary else generator._build_data_contract_text()
 requirements=hard_requirements(baseline_prompt)
 feature_payload='\n'+BINDING+'\n\n'+contract+'\n\nREQUIREMENTS REFERRED TO BY THIS TEMPLATE:\n'+requirements+'\n'
 output_reference='The initialization output is the complete executable function and its brief design rationale, serialized as specified above.'
 feature_payload+='\n'+output_reference+'\n'
 parent_payload='\n```python\n'+(parent or '')+'\n```\nDesign rationale: '+(explanation or 'No explanation available')+'\n'
 values=dict(features=feature_payload,parent=parent_payload,evaluation='\n'+(feedback or 'No performance metrics available')+'\n',violations='\n'+(feedback or 'The reported verifier failure must be corrected.')+'\n',syntactic='syntactic correctness (the exact executable interface and documented raw keys above)',mathematical='mathematical validity (finite outputs and guarded numerical operations as specified above)',semantic='semantic consistency (the unchanged decision-mode and verifier rules specified above)')
 return TEMPLATES[stage].format_map(values)

def install():
 from LLM_SR_DISTILLATION.llm_generator import LLMGenerator
 if getattr(LLMGenerator,'_paper_alignment_installed',False):return
 LLMGenerator._paper_alignment_installed=True
 names={'_build_initialization_prompt':'initialization','_build_evolution_mutation_prompt':'evolution','_build_constraint_fix_prompt':'repair','_build_constraint_retry_prompt':'repair'}
 for name,stage in names.items():
  original=getattr(LLMGenerator,name)
  def builder(self,*args,_original=original,_stage=stage,**kwargs):
   baseline=_original(self,*args,**kwargs)
   parent=kwargs.get('parent_function_code',args[0] if args else None)
   feedback=kwargs.get('parent_performance_info',kwargs.get('constraint_violations',args[1] if len(args)>1 else None))
   explanation=kwargs.get('parent_explanation',args[2] if len(args)>2 else None)
   return render(self,_stage,baseline,parent,explanation,feedback)
  setattr(LLMGenerator,name,builder)
