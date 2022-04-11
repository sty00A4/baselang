"""CONSTANTS"""
from typing import List
from string import ascii_letters as LETTERS
from string import digits as DIGITS
from sys import argv
LETTERS += "_"
VAR_CHARS = LETTERS + DIGITS
def string_with_arrows(text, pos_start, pos_end):
    result = ''

    # Calculate indices
    idx_start = max(text.rfind('\n', 0, pos_start.idx), 0)
    idx_end = text.find('\n', idx_start + 1)
    if idx_end < 0: idx_end = len(text)

    # Generate each line
    line_count = pos_end.ln - pos_start.ln + 1
    for i in range(line_count):
        # Calculate line columns
        line = text[idx_start:idx_end]
        col_start = pos_start.col if i == 0 else 0
        col_end = pos_end.col if i == line_count - 1 else len(line) - 1

        # Append to result
        result += line + '\n'
        result += ' ' * col_start + '^' * (col_end - col_start)

        # Re-calculate indices
        idx_start = idx_end
        idx_end = text.find('\n', idx_start + 1)
        if idx_end < 0: idx_end = len(text)

    return result.replace('\t', '')

"""POSITION"""
class Position:
    def __init__(self, idx, ln, col, fn, ftxt):
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt
    def next(self, char=None):
        self.idx += 1
        self.col += 1
        if char == "\n":
            self.ln += 1
            self.col = 0
        return self
    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)

"""ERROR"""
class Error:
    def __init__(self, pos_start: Position, pos_end: Position, error_name, details):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.name = error_name
        self.details = details
    def as_string(self):
        return f"{self.name}: {self.details}" + f"\nFile {self.pos_start.fn}, line {self.pos_start.ln + 1}" + f"\n{string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)}"
class IllagelCharError(Error):
    def __init__(self, pos_start, pos_end, details: str):
        super().__init__(pos_start, pos_end, "Illagel Character", details)
class ExpectedCharError(Error):
    def __init__(self, pos_start, pos_end, details: str):
        super().__init__(pos_start, pos_end, "Expected Character", details)
class InvalidSyntaxError(Error):
    def __init__(self, pos_start, pos_end, details: str):
        super().__init__(pos_start, pos_end, "Invalid Syntax", details)
class ConstantImmutableError(Error):
    def __init__(self, pos_start, pos_end, details: str):
        super().__init__(pos_start, pos_end, "Constant Immutable", details)
class UnimplementedError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, "Unimplemented", details)
class RTError(Error):
    def __init__(self, pos_start, pos_end, details, context):
        super().__init__(pos_start, pos_end, "Runtime Error", details)
        self.context = context
    def traceback(self):
        result = ""
        pos = self.pos_start
        ctx = self.context
        count = 0
        while ctx:
            result = f"file {self.pos_start.fn}, line {self.pos_start.ln + 1}, in {ctx.display_name}\n" + result
            pos = ctx.parent_entry_pos
            ctx = ctx.parent
            count += 1
        if count > 1:
            return "traceback (most recent call last):\n" + result
        else:
            return "traceback: " + result
    def as_string(self):
        return f"{self.traceback()}" \
               f"{self.name}: {self.details}\n" \
               f"{string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)}"

"""TOKENS"""
INT         = "INT"
FLOAT       = "FLOAT"
IDENTIFIER  = "IDENTIFIER"
KEYWORD     = "KEYWORD"
PLUS        = "PLUS"
MINUS       = "MINUS"
MUL         = "MUL"
DIV         = "DIV"
POW         = "POW"
EQ          = "EQ"
EVALIN      = "EVALIN"
EVALOUT     = "EVALOUT"
EE          = "EE"
NE          = "NE"
LT          = "LT"
GT          = "GT"
LTE         = "LTE"
GTE         = "GTE"
EOF         = "EOF"
KEYWORDS    = {
    "var_def": "let",
    "bool_and": "and",
    "bool_or": "or",
    "bool_not": "not"
}
class Token:
    def __init__(self, type_: str, value=None, pos_start=None, pos_end=None):
        self.type = type_
        self.value = value
        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.next()
        if pos_end:
            self.pos_end = pos_end.copy()
    def matches(self, type_, value):
        return self.type == type_ and self.value == value
    def __repr__(self):
        if self.value: return f"[{self.type}:{self.value}]"
        return f"[{self.type}]"

"""LEXER""" #1
class Lexer:
    def __init__(self, fn: str, text: str):
        self.fn = fn
        self.text = text
        self.pos = Position(-1, 0, -1, self.fn, self.text)
        self.char = None
        self.next()
    def next(self):
        self.pos.next(self.char)
        self.char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None
    def make_tokens(self):
        tokens = []
        while self.char is not None:
            if self.char in " \t": # space
                self.next()
            elif self.char in DIGITS: # numbers
                tokens.append(self.make_num())
            elif self.char in LETTERS: # variable or keyword
                tokens.append(self.make_id())
            elif self.char == "=": # plus
                tokens.append(self.make_equals())
                self.next()
            elif self.char == "<": # plus
                tokens.append(self.make_lt())
                self.next()
            elif self.char == ">": # plus
                tokens.append(self.make_gt())
                self.next()
            elif self.char == "+": # plus
                tokens.append(Token(PLUS, pos_start=self.pos.copy()))
                self.next()
            elif self.char == "-": # minus
                tokens.append(Token(MINUS, pos_start=self.pos.copy()))
                self.next()
            elif self.char == "*": # multiply
                tokens.append(Token(MUL, pos_start=self.pos.copy()))
                self.next()
                if self.char == "*": # power
                    tokens[-1] = Token(POW, pos_start=self.pos.copy())
                    self.next()
            elif self.char == "/": # divide
                tokens.append(Token(DIV, pos_start=self.pos.copy()))
                self.next()
            elif self.char == "(": # evaluation in
                tokens.append(Token(EVALIN, pos_start=self.pos.copy()))
                self.next()
            elif self.char == ")": # evaluation out
                tokens.append(Token(EVALOUT, pos_start=self.pos.copy()))
                self.next()
            elif self.char == "!":  # not equal
                tok, error = self.make_ne()
                if error: return [], error
                tokens.append(tok)
                self.next()
            else:
                pos_start = self.pos.copy()
                char = self.char
                self.next()
                return [], IllagelCharError(pos_start, self.pos, f"'{char}'")
        tokens.append(Token(EOF, pos_start=self.pos.copy()))
        return tokens, None
    def make_num(self):
        num_str = ""
        dot_count = 0
        pos_start = self.pos.copy()
        while self.char is not None and self.char in DIGITS + ".":
            if self.char == ".":
                if dot_count == 1: break
                dot_count += 1
                num_str += "."
            else:
                num_str += self.char
            self.next()
        if dot_count == 0:
            return Token(INT, int(num_str), pos_start, self.pos.copy())
        else:
            return Token(FLOAT, float(num_str), pos_start, self.pos.copy())
    def make_id(self):
        id_str = ""
        pos_start = self.pos.copy()
        while self.char is not None and self.char in VAR_CHARS:
            id_str += self.char
            self.next()
        tok_type = KEYWORD if id_str in KEYWORDS.values() else IDENTIFIER
        return Token(tok_type, id_str, pos_start, self.pos)
    def make_ne(self):
        pos_start = self.pos.copy()
        self.next()
        if self.char == "=":
            return Token(NE, pos_start=pos_start.copy(), pos_end=self.pos.copy()), None
        self.next()
        return None, ExpectedCharError(
            pos_start.copy(), self.pos.copy(),
            "'=' after '!'"
        )
    def make_equals(self):
        tok_type = EQ
        pos_start = self.pos.copy()
        self.next()
        if self.char == "=":
            tok_type = EE
        return Token(tok_type, pos_start=pos_start.copy(), pos_end=self.pos.copy())
    def make_lt(self):
        tok_type = LT
        pos_start = self.pos.copy()
        self.next()
        if self.char == "=":
            tok_type = LTE
        return Token(tok_type, pos_start=pos_start.copy(), pos_end=self.pos.copy())
    def make_gt(self):
        tok_type = GT
        pos_start = self.pos.copy()
        self.next()
        if self.char == "=":
            tok_type = GTE
        return Token(tok_type, pos_start=pos_start.copy(), pos_end=self.pos.copy())

"""NODES"""
class NumberNode:
    def __init__(self, tok):
        self.tok = tok
        self.pos_start = tok.pos_start.copy()
        self.pos_end = tok.pos_end.copy()
    def __repr__(self):
        return f"{self.tok}"
class VarAccessNode:
    def __init__(self, var_name_tok):
        self.var_name_tok = var_name_tok
        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.var_name_tok.pos_end
class VarAssignNode:
    def __init__(self, var_name_tok, value_node):
        self.var_name_tok = var_name_tok
        self.value_node = value_node
        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.value_node.pos_end
class BinOpNode:
    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node
        self.pos_start = left_node.pos_start.copy()
        self.pos_end = right_node.pos_end.copy()
    def __repr__(self):
        return f"({self.left_node} {self.op_tok} {self.right_node})"
class UnaryOpNode:
    def __init__(self, op_tok, node):
        self.op_tok = op_tok
        self.node = node
        self.pos_start = op_tok.pos_start.copy()
        self.pos_end = node.pos_end.copy()
    def __repr__(self):
        return f"(u{self.op_tok} {self.node})"

"""PARSE RESULT"""
class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
        self.next_count = 0
    def register_next(self):
        self.next_count += 1
    def register(self, res):
        self.next_count += res.next_count
        if res.error: self.error = res.error
        return res.node
    def success(self, node):
        self.node = node
        return self
    def failure(self, error):
        if not self.error or self.next_count == 0:
            self.error = error
        return self

"""PARSER""" #2
class Parser:
    def __init__(self, tokens):
        self.tok = None
        self.tokens = tokens
        self.tok_idx = -1
        self.next()
    def next(self):
        self.tok_idx += 1
        if self.tok_idx < len(self.tokens):
            self.tok = self.tokens[self.tok_idx]
        return self.tok

    def parse(self):
        res = self.expr()
        if not res.error and self.tok.type != EOF:
            return res.failure(InvalidSyntaxError(
                self.tok.pos_start, self.tok.pos_end,
                "expected '+', '-', '*', '/'"
            ))
        return res

    def atom(self):
        res = ParseResult()
        tok = self.tok
        if tok.type in (INT, FLOAT): # factor NumberNode
            res.register_next()
            self.next()
            return res.success(NumberNode(tok))
        elif tok.type == IDENTIFIER:
            res.register_next()
            self.next()
            return res.success(VarAccessNode(tok))
        elif tok.type == EVALIN: # factor EVALIN
            res.register_next()
            self.next()
            expr = res.register(self.expr())
            if res.error: return res
            if self.tok.type == EVALOUT:
                res.register_next()
                self.next()
                return res.success(expr)
            else: return res.failure(InvalidSyntaxError(tok.pos_start, tok.pos_end, "missing ')'")) # factor InvalidSyntaxError missing ')'
        return res.failure(InvalidSyntaxError(
            tok.pos_start, tok.pos_end,
            "expected int, float, identifier, '+', '-', '('"
        ))

    def power(self):
        return self.bin_op(self.atom, [POW], self.factor)

    def factor(self):
        res = ParseResult()
        tok = self.tok
        if tok.type in (PLUS, MINUS): # factor UnaryOpNode
            res.register_next()
            self.next()
            factor = res.register(self.factor())
            if res.error: return res
            return res.success(UnaryOpNode(tok, factor))
        return self.power()
    def term(self):
        return self.bin_op(self.factor, [MUL, DIV])
    def expr(self):
        res = ParseResult()
        if self.tok.matches(KEYWORD, KEYWORDS["var_def"]):
            res.register_next()
            self.next()
            if self.tok.type != IDENTIFIER: return res.failure(InvalidSyntaxError(
                self.tok.pos_start, self.tok.pos_end,
                "expected identifier"
            ))
            var_name = self.tok
            res.register_next()
            self.next()
            if self.tok.type != EQ: return res.failure(InvalidSyntaxError(
                self.tok.pos_start, self.tok.pos_end,
                "expected '='"
            ))
            res.register_next()
            self.next()
            expr = res.register(self.expr())
            if res.error: return res
            return res.success(VarAssignNode(var_name, expr))
        node = res.register(self.bin_op(self.term, [PLUS, MINUS]))
        if res.error:
            return res.failure(InvalidSyntaxError(
                self.tok.pos_start, self.tok.pos_end,
                f"expected int, float, identifier, '+', '-', '(', '{KEYWORDS['var_def']}'"
            ))
        return res.success(node)
    def bin_op(self, func1, ops: list, func2=None):
        if func2 is None:
            func2 = func1
        res = ParseResult()
        left = res.register(func1())
        if res.error: return res
        while self.tok.type in ops:
            op_tok = self.tok
            res.register_next()
            self.next()
            right = res.register(func2())
            if res.error: return res
            left = BinOpNode(left, op_tok, right)
        return res.success(left)

"""RUNTIME RESULT"""
class RTResult:
    def __init__(self):
        self.value = None
        self.error = None
    def register(self, res):
        if res.error: self.error = res.error
        return res.value
    def success(self, value):
        self.value = value
        return self
    def failure(self, error):
        self.error = error
        return self

"""VALUES"""
class Number:
    def __init__(self, value):
        self.value = value
        self.set_pos()
        self.set_context()
    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self
    def set_context(self, context=None):
        self.context = context
        return self
    def copy(self):
        return Number(self.value)

    def add(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.context), None
    def sub(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None
    def mul(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context), None
    def div(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(other.pos_start, other.pos_end, "division by zero", self.context)
            return Number(self.value / other.value).set_context(self.context), None
    def pow(self, other):
        if isinstance(other, Number):
            return Number(self.value ** other.value).set_context(self.context), None
    def neg(self):
        return Number(-self.value).set_context(self.context), None

    def __repr__(self):
        return f"{self.value}"

"""CONTEXT"""
class Context:
    def __init__(self, display_name, parent=None, parent_entry_pos=None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.vars = None

"""SYMBOL TABLE"""
class Vars:
    def __init__(self):
        self.vars = {}
        self.parent = None
    def get(self, name):
        value = self.vars.get(name, None)
        if value is None and self.parent: return self.parent.get(name)
        return value
    def set(self, name, value):
        self.vars[name] = value
    def remove(self, name):
        del self.vars[name]

"""INTERPRETER""" #3
class Interpreter:
    def visit(self, node, context):
        method_name = f"visit_{type(node).__name__}"
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)
    def no_visit_method(self, node, context):
        raise Exception(f"no visit_{type(node).__name__} method defined")

    def visit_NumberNode(self, node, context): # number
        return RTResult().success(Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end))
    def visit_VarAccessNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = context.vars.get(var_name)
        if not value:
            return res.failure(RTError(
                node.pos_start, node.pos_end,
                f"'{var_name}' is not defined",
                context
            ))
        value = value.copy().set_pos(node.pos_start, node.pos_end)
        return res.success(value)
    def visit_VarAssignNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = res.register(self.visit(node.value_node, context))
        if res.error: return res
        context.vars.set(var_name, value)
        return res.success(value)
    def visit_BinOpNode(self, node, context): # bin op
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.error: return res
        right = res.register(self.visit(node.right_node, context))
        if res.error: return res
        if node.op_tok.type == PLUS: # add
            result, error = left.add(right)
        if node.op_tok.type == MINUS: # sub
            result, error = left.sub(right)
        if node.op_tok.type == MUL: # mul
            result, error = left.mul(right)
        if node.op_tok.type == DIV: # div
            result, error = left.div(right)
        if node.op_tok.type == POW: # pow
            result, error = left.pow(right)
        if error: return res.failure(error)
        return res.success(result.set_pos(node.pos_start, node.pos_end))
    def visit_UnaryOpNode(self, node, context): # unary op
        res = RTResult()
        number = res.register(self.visit(node.node, context))
        if res.error: return res
        if node.op_tok.type == MINUS:
            number, error = number.neg()
        if error: return res.failure(error)
        return res.success(number.set_pos(node.pos_start, node.pos_end))

"""RUN"""
global_vars = Vars()
global_vars.set("null", Number(0))
global_vars.set("true", Number(1))
global_vars.set("false", Number(0))
def run(fn: str, text: str):
    lexer = Lexer(fn, text)
    tokens, error = lexer.make_tokens()
    if error: return None, error

    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: return None, ast.error

    interperter = Interpreter()
    context = Context("<program>")
    context.vars = global_vars
    result = interperter.visit(ast.node, context)

    return result.value, result.error

if len(argv) > 1:
    with open(argv[1], "r") as f:
        res, error = run(argv[1], f.read())
        if error:
            print(error.as_string())
        print(res)