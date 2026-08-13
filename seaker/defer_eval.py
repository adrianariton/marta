import ast
import inspect
import textwrap


def Do_something():
    print("!!! LOGIC RUNNING BEFORE EVALUATION !!!")


class MyClass:
    def __init__(self, p1, p2):
        print(f"Class Init with: {p1}, {p2}")


DEFERABLE_CLASSES = ["While"]


def preprocess(func):
    # 1. Get the source code of the decorated function
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)

    # 2. Walk the tree and find MyClass(args)
    class RewriteCall(ast.NodeTransformer):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id in DEFERABLE_CLASSES:
                new_node = ast.Subscript(
                    value=ast.Tuple(
                        elts=[
                            ast.Call(
                                func=ast.Name(id="Do_something", ctx=ast.Load()),
                                args=[],
                                keywords=[],
                            ),
                            node,
                        ],
                        ctx=ast.Load(),
                    ),
                    slice=ast.Constant(value=-1),
                    ctx=ast.Load(),
                )
                return ast.copy_location(new_node, node)
            return node

    # 3. Apply transformation and compile
    new_tree = RewriteCall().visit(tree)
    ast.fix_missing_locations(new_tree)

    # Remove the decorator from the rewritten code to avoid infinite recursion
    new_tree.body[0].decorator_list = []

    code_obj = compile(new_tree, filename="<ast>", mode="exec")
    namespace = func.__globals__
    exec(code_obj, namespace)
    return namespace[func.__name__]


# --- USAGE ---


@preprocess
def run_logic():
    print("Starting function...")
    # These parameters would normally evaluate BEFORE anything else
    obj = MyClass(print("Eval P1") or 1, print("Eval P2") or 2)


run_logic()
