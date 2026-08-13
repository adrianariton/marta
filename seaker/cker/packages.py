from cker.program import _get_program, Variable


def load_torch() -> Variable:
    program = _get_program()
    return program.package("torch")


def load_transformers() -> Variable:
    program = _get_program()
    return program.package("transformers")


def load_sentence_transformers() -> Variable:
    program = _get_program()
    return program.package("sentence_transformers")


def load_spacy() -> Variable:
    program = _get_program()
    return program.package("spacy")


def load_sklearn() -> Variable:
    program = _get_program()
    return program.package("sklearn")


def load_pydantic() -> Variable:
    program = _get_program()
    return program.package("pydantic")
