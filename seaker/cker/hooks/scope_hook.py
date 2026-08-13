import sys
import ast
import inspect
import textwrap
from cker.program import While, If, Else, ElseIf


class ScopeTransformer(ast.NodeTransformer):
    def visit_If(self, node):
        """
        We transform:
        if x > 5:
            body
        elif x > 4:
            body2
        else:
            body3

        Into:
        with If(x > 5):
            body
        with Else():
            with If(x > 4):
                body2
            with Else():
                body3
        """
        self.generic_visit(node)
        new_orelse = node.orelse
        if node.orelse:
            new_orelse = [
                ast.With(
                    items=[
                        ast.withitem(
                            context_expr=ast.Call(
                                func=ast.Name(id="Else", ctx=ast.Load()), args=[], keywords=[]
                            )
                        )
                    ],
                    body=node.orelse,
                )
            ]
        return ast.With(
            items=[
                ast.withitem(
                    context_expr=ast.Call(
                        func=ast.Name(id="If", ctx=ast.Load()), args=[node.test], keywords=[]
                    ),
                    optional_vars=ast.Name(id="_run", ctx=ast.Store()),
                )
            ],
            body=[
                ast.If(test=ast.Name(id="_run", ctx=ast.Load()), body=node.body, orelse=new_orelse)
            ],
        )

    def visit_While(self, node):
        """
        We transform:
        while x > 5:
            body

        Into:
        with While(x > 5):
            body
            # (No loop here, just the body)
        """
        self.generic_visit(node)
        return ast.With(
            items=[
                ast.withitem(
                    context_expr=ast.Call(
                        func=ast.Name(id="While", ctx=ast.Load()), args=[node.test], keywords=[]
                    )
                )
            ],
            body=node.body,
        )


class ScopeHook:
    _backups = {}
    active = False  # The Master Switch

    def start(self):
        frame = sys._getframe(1)
        func_obj = frame.f_globals[frame.f_code.co_name]
        if hasattr(func_obj, "_patched"):
            return

        ScopeHook.active = True
        self._backups[id(func_obj)] = func_obj.__code__

        source = textwrap.dedent(inspect.getsource(func_obj))
        tree = ast.parse(source)
        func_def = tree.body[0]
        func_def.body = [
            n
            for n in func_def.body
            if not (
                isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Call)
                and ("start" in ast.dump(n) or 'print("1. Patching...")' in ast.dump(n))
            )
        ]

        transformed = ScopeTransformer().visit(tree)
        ast.fix_missing_locations(transformed)

        func_obj.__code__ = compile(transformed, "<transformed>", "exec").co_consts[0]
        func_obj._patched = True
        func_obj.__globals__.update(frame.f_locals)
        func_obj.__globals__.update({"If": If, "While": While, "Else": Else, "hook": self})

        try:
            func_obj()
        finally:
            sys.exit(0)

    def end(self):
        ScopeHook.active = False  # Kill the logging immediately
        frame = sys._getframe(1)
        func_obj = frame.f_globals.get(frame.f_code.co_name)
        if func_obj and id(func_obj) in self._backups:
            print("\n--- [SURGERY COMPLETE: REVERTING] ---")
            func_obj.__code__ = self._backups[id(func_obj)]
            if hasattr(func_obj, "_patched"):
                delattr(func_obj, "_patched")
