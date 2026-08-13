from cker.program import _get_program, Variable


def load_hf_model(model_name: str) -> Variable:
    program = _get_program()
    return program.execute("load_hf_model", model_name)
