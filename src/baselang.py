"""CONSTANTS"""
from string import ascii_letters as LETTERS
from string import digits as DIGITS
from sys import argv
from time import sleep, time
from math import pi
LETTERS += "_"
VAR_CHARS = LETTERS + DIGITS
def string_with_arrows(text, pos_start, pos_end):
    result = ''
    idx_start = max(text.rfind('\n', 0, pos_start.idx), 0)
    idx_end = text.find('\n', idx_start + 1)
    if idx_end < 0: idx_end = len(text)
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
class UserDefinedError(Error):
    def __init__(self, name, details):
        super().__init__(None, None, name, details)
    def as_string(self):
        return f"{self.name}: {self.details}"
# lexer
class IllagelCharError(Error):
    def __init__(self, pos_start, pos_end, details: str):
        super().__init__(pos_start, pos_end, "Illagel Character", details)
class ExpectedCharError(Error):
    def __init__(self, pos_start, pos_end, details: str):
        super().__init__(pos_start, pos_end, "Expected Character", details)
# parser
class InvalidSyntaxError(Error):
    def __init__(self, pos_start, pos_end, details: str):
        super().__init__(pos_start, pos_end, "Invalid Syntax", details)
# run time
class RTError(Error):
    def __init__(self, pos_start, pos_end, details, context):
        super().__init__(pos_start, pos_end, "Runtime Error", details)
        self.context = context
    def traceback(self):
        result = ""
        ctx = self.context
        count = 0
        while ctx:
            result = f"file {self.pos_start.fn}, line {self.pos_start.ln + 1}, in {ctx.display_name}\n" + result
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
class ConstantImmutableError(Error):
    def __init__(self, pos_start, pos_end, details: str):
        super().__init__(pos_start, pos_end, "Constant Immutable", details)
class UnimplementedError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, "Unimplemented", details)

"""TOKENS"""
NUMBER      = "NUMBER"
STRING      = "STRING"
NULL        = "NULL"
TRUE        = "TRUE"
FALSE       = "FALSE"
IDENTIFIER  = "IDENTIFIER"
KEYWORD     = "KEYWORD"
PLUS        = "PLUS"
MINUS       = "MINUS"
MUL         = "MUL"
DIV         = "DIV"
DIVINT      = "DIVINT"
POW         = "POW"
INDEX       = "INDEX"
MOD         = "MOD"
EQ          = "EQ"
EVALIN      = "EVALIN"
EVALOUT     = "EVALOUT"
LISTIN      = "LISTIN"
LISTOUT     = "LISTOUT"
TABLEIN     = "TABLEIN"
TABLEOUT    = "TABLEOUT"
EE          = "EE"
NE          = "NE"
LT          = "LT"
GT          = "GT"
LTE         = "LTE"
GTE         = "GTE"
IN          = "IN"
INC         = "INC"
DEC         = "DEC"
SEP         = "SEP"
REP         = "REP"
NL          = "NL"
EOF         = "EOF"
KEYWORDS    = {
    "null":     "null",
    "true":     "true",
    "false":    "false",
    "bool_and": "and",
    "bool_or":  "or",
    "bool_not": "not",
    "in":       "in",
    "if":       "if",
    "then":     "then",
    "elif":     "elif",
    "else":     "else",
    "for":      "for",
    "forEvery": "forEvery",
    "of":       "of",
    "step":     "step",
    "do":       "do",
    "while":    "while",
    "function": "func",
    "end":      "end",
    "break":    "break",
    "next":     "next",
    "return":   "return",
    "use":      "use",
}
INVALIDSYNTAX_START = "number, null, bool, string, identifier, '+', '-', '(', '[', '{', " + f"'{KEYWORDS['if']}', '{KEYWORDS['for']}', '{KEYWORDS['while']}', '{KEYWORDS['function']}', '{KEYWORDS['bool_not']}'"
INVALIDSYNTAX_ALL = "number, null, bool, string, identifier, '+', '-', '*', '/', '**' '(', '[', '{', " + f"'{KEYWORDS['if']}', '{KEYWORDS['for']}', '{KEYWORDS['while']}', '{KEYWORDS['function']}, '{KEYWORDS['bool_not']}'"
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
    def matches(self, type_, value=None):
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
            elif self.char in "'": # comment
                self.next()
                while not self.char in "'":
                    self.next()
                self.next()
            elif self.char in ";\n": # new line
                tokens.append(Token(NL, pos_start=self.pos.copy()))
                self.next()
            elif self.char in DIGITS: # numbers
                tokens.append(self.make_num())
            elif self.char in LETTERS: # variable or keyword
                tokens.append(self.make_id())
            elif self.char == '"': # string
                tokens.append(self.make_str())
            elif self.char == "#": # index
                tokens.append(Token(INDEX, pos_start=self.pos.copy()))
                self.next()
            elif self.char == "%": # mod
                tokens.append(Token(MOD, pos_start=self.pos.copy()))
                self.next()
            elif self.char == "=": # equal
                tokens.append(self.make_equals())
                self.next()
            elif self.char == "!":  # not equal
                tok, error = self.make_ne()
                if error: return [], error
                tokens.append(tok)
                self.next()
            elif self.char == "<": # lt
                tokens.append(self.make_lt())
                self.next()
            elif self.char == ">": # gt
                tokens.append(self.make_gt())
                self.next()
            elif self.char == "+": # plus
                pos_start = self.pos.copy()
                tokens.append(Token(PLUS, pos_start=pos_start))
                self.next()
                if self.char == "+": # inc
                    tokens[-1] = Token(INC, pos_start=pos_start, pos_end=self.pos.copy())
                    self.next()
            elif self.char == "-": # minus
                pos_start = self.pos.copy()
                tokens.append(Token(MINUS, pos_start=self.pos.copy()))
                self.next()
                if self.char == "-": # dec
                    tokens[-1] = Token(DEC, pos_start=pos_start, pos_end=self.pos.copy())
                    self.next()
            elif self.char == "*": # multiply
                pos_start = self.pos.copy()
                tokens.append(Token(MUL, pos_start=pos_start))
                self.next()
                if self.char == "*": # power
                    tokens[-1] = Token(POW, pos_start=pos_start, pos_end=self.pos.copy())
                    self.next()
            elif self.char == "/": # divide
                pos_start = self.pos.copy()
                tokens.append(Token(DIV, pos_start=self.pos.copy()))
                self.next()
                if self.char == "/": # int divide
                    tokens[-1] = Token(DIVINT, pos_start=pos_start, pos_end=self.pos.copy())
                    self.next()
            elif self.char == "(": # evaluation in
                tokens.append(Token(EVALIN, pos_start=self.pos.copy()))
                self.next()
            elif self.char == ")": # evaluation out
                tokens.append(Token(EVALOUT, pos_start=self.pos.copy()))
                self.next()
            elif self.char == "[": # list in
                tokens.append(Token(LISTIN, pos_start=self.pos.copy()))
                self.next()
            elif self.char == "]": # list out
                tokens.append(Token(LISTOUT, pos_start=self.pos.copy()))
                self.next()
            elif self.char == "{": # list in
                tokens.append(Token(TABLEIN, pos_start=self.pos.copy()))
                self.next()
            elif self.char == "}": # list out
                tokens.append(Token(TABLEOUT, pos_start=self.pos.copy()))
                self.next()
            elif self.char == ",":  # sep
                tokens.append(Token(SEP, pos_start=self.pos.copy()))
                self.next()
            elif self.char == ":":  # rep
                tokens.append(Token(REP, pos_start=self.pos.copy()))
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
        if dot_count == 0: return Token(NUMBER, int(num_str), pos_start, self.pos.copy())
        else: return Token(NUMBER, float(num_str), pos_start, self.pos.copy())
    def make_id(self):
        id_str = ""
        pos_start = self.pos.copy()
        while self.char is not None and self.char in VAR_CHARS:
            id_str += self.char
            self.next()
        tok_type = KEYWORD if id_str in KEYWORDS.values() else IDENTIFIER
        tok_type = IN if id_str == KEYWORDS["in"] else tok_type
        tok_type = NULL if id_str == KEYWORDS["null"] else tok_type
        tok_type = TRUE if id_str == KEYWORDS["true"] else tok_type
        tok_type = FALSE if id_str == KEYWORDS["false"] else tok_type
        if tok_type in (KEYWORD, IDENTIFIER): return Token(tok_type, id_str, pos_start, self.pos.copy())
        else: return Token(tok_type, None, pos_start, self.pos.copy())
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
    def make_str(self):
        string = ""
        pos_start = self.pos.copy()
        escape_character = False
        self.next()
        replace = {
            "n": "\n",
            "t": "\t",
        }
        while self.char is not None and (self.char != '"' or escape_character):
            if escape_character:
                string += replace.get(self.char, self.char)
            else:
                if self.char == "\\":
                    escape_character = True
                else:
                    string += self.char
            self.next()
            escape_character = False
        self.next()
        return Token(STRING, string, pos_start, self.pos.copy())

"""NODES"""
class NumberNode:
    def __init__(self, tok):
        self.tok        = tok
        self.pos_start  = tok.pos_start.copy()
        self.pos_end    = tok.pos_end.copy()
    def __repr__(self):
        return f"(number {self.tok})"
class NullNode:
    def __init__(self, tok):
        self.tok        = tok
        self.pos_start  = tok.pos_start.copy()
        self.pos_end    = tok.pos_end.copy()
    def __repr__(self):
        return f"(null)"
class BoolNode:
    def __init__(self, tok):
        self.tok        = tok
        self.pos_start  = tok.pos_start.copy()
        self.pos_end    = tok.pos_end.copy()
    def __repr__(self):
        return f"(bool {self.tok})"
class StringNode:
    def __init__(self, tok):
        self.tok        = tok
        self.pos_start  = tok.pos_start.copy()
        self.pos_end    = tok.pos_end.copy()
    def __repr__(self):
        return f"(string \"{self.tok}\")"
class ListNode:
    def __init__(self, element_nodes, pos_start, pos_end):
        self.element_nodes = element_nodes
        self.pos_start = pos_start
        self.pos_end = pos_end
class TableNode:
    def __init__(self, element_nodes, key_nodes, pos_start, pos_end):
        self.element_nodes = element_nodes
        self.key_nodes = key_nodes
        self.pos_start = pos_start
        self.pos_end = pos_end
class VarAccessNode:
    def __init__(self, var_name_tok):
        self.var_name_tok   = var_name_tok
        self.pos_start      = self.var_name_tok.pos_start
        self.pos_end        = self.var_name_tok.pos_end
    def __repr__(self):
        return f"(varAccessNode {self.var_name_tok})"
class VarAssignNode:
    def __init__(self, var_name_tok, value_node=None):
        self.var_name_tok           = var_name_tok
        self.value_node             = value_node
        self.pos_start              = self.var_name_tok.pos_start
        if value_node: self.pos_end = self.value_node.pos_end
        else: self.pos_end          = self.var_name_tok.pos_end
class BinOpNode:
    def __init__(self, left_node, op_tok, right_node):
        self.left_node  = left_node
        self.op_tok     = op_tok
        self.right_node = right_node
        self.pos_start  = left_node.pos_start.copy()
        self.pos_end    = right_node.pos_end.copy()
    def __repr__(self):
        return f"({self.left_node} {self.op_tok} {self.right_node})"
class UnaryOpNode:
    def __init__(self, op_tok, node):
        self.op_tok     = op_tok
        self.node       = node
        self.pos_start  = op_tok.pos_start.copy()
        self.pos_end    = node.pos_end.copy()
    def __repr__(self):
        return f"(u{self.op_tok} {self.node})"
class VarIncrementationNode:
    def __init__(self, var_name_tok):
        self.var_name_tok   = var_name_tok
        self.pos_start      = self.var_name_tok.pos_start
        self.pos_end        = self.var_name_tok.pos_end
class IfNode:
    def __init__(self, cases, else_case=None, return_null=False):
        self.cases      = cases
        self.else_case  = else_case
        self.return_null = return_null
        self.pos_start  = self.cases[0][0].pos_start
        self.pos_end    = (self.else_case or self.cases[len(self.cases) - 1])[0].pos_end
    def __repr__(self):
        nl = "\n\t"
        return f"(ifNode cases({nl.join(str(x) for x in self.cases)}) else({self.else_case}) return_null({self.return_null}) ({self.pos_start}-{self.pos_end}))"
class ForNode:
    def __init__(self, var_name_tok, start_value_node, end_value_node, step_value_node, body_node, return_null=False):
        self.body_node          = body_node
        self.step_value_node    = step_value_node
        self.end_value_node     = end_value_node
        self.start_value_node   = start_value_node
        self.var_name_tok       = var_name_tok
        self.return_null = return_null

        self.pos_start  = self.var_name_tok.pos_start
        self.pos_end    = self.body_node.pos_end
class ForEveryNode:
    def __init__(self, var_name_tok, list_tok, body_node, return_null=False):
        self.body_node          = body_node
        self.list_name_tok      = list_tok
        self.var_name_tok       = var_name_tok
        self.return_null = return_null

        self.pos_start  = self.var_name_tok.pos_start
        self.pos_end    = self.body_node.pos_end
class WhileNode:
    def __init__(self, condition_node, body_node, return_null=False):
        self.condition_node = condition_node
        self.body_node      = body_node
        self.return_null = return_null

        self.pos_start  = self.condition_node.pos_start
        self.pos_end    = self.body_node.pos_end
class FuncDefNode:
    def __init__(self, var_name_tok, arg_name_toks, body_node, auto_return=False):
        self.var_name_tok = var_name_tok
        self.arg_name_toks = arg_name_toks
        self.body_node = body_node
        self.auto_return = auto_return
        if self.var_name_tok: self.pos_start = self.var_name_tok.pos_start
        elif len(self.arg_name_toks) > 0: self.pos_start = self.arg_name_toks[0].pos_start
        else: self.pos_start = self.body_node.pos_start
        self.pos_end = self.body_node.pos_end
class CallNode:
    def __init__(self, node_to_call: VarAccessNode, arg_nodes):
        self.node_to_call = node_to_call
        self.arg_nodes = arg_nodes
        self.pos_start = self.node_to_call.pos_start
        if len(self.arg_nodes) > 0: self.pos_end = self.arg_nodes[-1].pos_end
        else: self.pos_end = self.node_to_call.pos_end
class ReturnNode:
    def __init__(self, node_to_return, pos_start, pos_end):
        self.node_to_return = node_to_return
        self.pos_start = pos_start
        self.pos_end = pos_end
class NextNode:
    def __init__(self, pos_start, pos_end):
        self.pos_start = pos_start
        self.pos_end = pos_end
class BreakNode:
    def __init__(self, pos_start, pos_end):
        self.pos_start = pos_start
        self.pos_end = pos_end
class UseNode:
    def __init__(self, file_name_tok):
        self.file_name_tok = file_name_tok
        self.pos_start = file_name_tok.pos_start.copy()
        self.pos_end = file_name_tok.pos_end.copy()

"""PARSE RESULT"""
class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
        self.last_registered_next_count = 0
        self.next_count = 0
        self.to_reverse_count = 0
    def register_next(self):
        self.last_registered_next_count = 1
        self.next_count += 1
    def register(self, res):
        self.last_registered_next_count = res.next_count
        self.next_count += res.next_count
        if res.error: self.error = res.error
        return res.node
    def try_register(self, res):
        if res.error:
            self.to_reverse_count = res.next_count
            return None
        return self.register(res)
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
    def reverse(self, amount=1):
        self.tok_idx -= amount
        self.update_tok()
        return self.tok
    def update_tok(self):
        if self.tok_idx >= 0 and self.tok_idx < len(self.tokens):
            self.tok = self.tokens[self.tok_idx]
    def parse(self):
        res = self.statements()
        if not res.error and self.tok.type != EOF:
            return res.failure(InvalidSyntaxError(
                self.tok.pos_start, self.tok.pos_end,
                f"expected '+', '-', '*', '/', '**', '==', '!=', '<', '>', <=', '>=', '{KEYWORDS['bool_and']}', '{KEYWORDS['bool_or']}'"
            ))
        return res

    def next(self):
        self.tok_idx += 1
        self.update_tok()
        return self.tok
    def func_def(self):
        res = ParseResult()
        if not self.tok.matches(KEYWORD, KEYWORDS["function"]):
            return res.failure(InvalidSyntaxError(
                self.tok.pos_start, self.tok.pos_end,
                f"expected '{KEYWORDS['func']}'"
            ))
        res.register_next()
        self.next()
        if self.tok.type == IDENTIFIER:
            var_name_tok = self.tok
            res.register_next()
            self.next()
            if self.tok.type != EVALIN:
                return res.failure(InvalidSyntaxError(
                    self.tok.pos_start, self.tok.pos_end,
                    f"expected '('"
                ))
        else:
            var_name_tok = None
            if self.tok.type != EVALIN:
                return res.failure(InvalidSyntaxError(
                    self.tok.pos_start, self.tok.pos_end,
                    f"expected identifier, '('"
                ))
        res.register_next()
        self.next()
        arg_name_toks = []
        if self.tok.type == IDENTIFIER:
            arg_name_toks.append(self.tok)
            res.register_next()
            self.next()
            while self.tok.type == SEP:
                res.register_next()
                self.next()
                if self.tok.type != IDENTIFIER:
                    return res.failure(InvalidSyntaxError(
                        self.tok.pos_start, self.tok.pos_end,
                        f"expected identifier"
                    ))
                arg_name_toks.append(self.tok)
                res.register_next()
                self.next()
            if self.tok.type != EVALOUT:
                return res.failure(InvalidSyntaxError(
                    self.tok.pos_start, self.tok.pos_end,
                    f"expected ',', ')'"
                ))
        else:
            if self.tok.type != EVALOUT:
                return res.failure(InvalidSyntaxError(
                    self.tok.pos_start, self.tok.pos_end,
                    f"expected identifier, ')'"
                ))
        res.register_next()
        self.next()
        if self.tok.type == REP:
            res.register_next()
            self.next()
            node_to_return = res.register(self.expr())
            if res.error: return res
            return res.success(FuncDefNode(var_name_tok, arg_name_toks, node_to_return, True))
        if self.tok.type != NL:
            return res.failure(InvalidSyntaxError(
                self.tok.pos_start, self.tok.pos_end,
                f"expected ':' or a new line"
            ))
        res.register_next()
        self.next()
        body = res.register(self.statements())
        if res.error: return res
        if not self.tok.matches(KEYWORD, KEYWORDS["end"]):
            return res.failure(InvalidSyntaxError(
                self.tok.pos_start, self.tok.pos_end,
                f"expected '{KEYWORDS['end']}'"
            ))
        res.register_next()
        self.next()
        return res.success(FuncDefNode(var_name_tok, arg_name_toks, body))
    def while_expr(self):
        res = ParseResult()
        if not self.tok.matches(KEYWORD, KEYWORDS['while']):
            return res.failure(InvalidSyntaxError(
                self.tok.pos_start, self.tok.pos_end,
                f"expected '{KEYWORDS['while']}'"
            ))
        res.register_next()
        self.next()
        condition = res.register(self.expr())
        if res.error: return res
        if self.tok.matches(KEYWORD, KEYWORDS['do']):
            res.register_next()
            self.next()
            if self.tok.type == NL:
                res.register_next()
                self.next()
                body = res.register(self.statements())
                if res.error: return res
                if not self.tok.matches(KEYWORD, KEYWORDS["end"]):
                    return res.failure(InvalidSyntaxError(
                        self.tok.pos_start, self.tok.pos_end,
                        f"expected '{KEYWORDS['end']}'"
                    ))
                res.register_next()
                self.next()
                return res.success(WhileNode(condition, body, True))
            body = res.register(self.statement())
            if res.error: return res
            return res.success(WhileNode(condition, body))
        if self.tok.type == NL:
            res.register_next()
            self.next()
            body = res.register(self.statements())
            if res.error: return res
            if not self.tok.matches(KEYWORD, KEYWORDS["end"]):
                return res.failure(InvalidSyntaxError(
                    self.tok.pos_start, self.tok.pos_end,
                    f"expected '{KEYWORDS['end']}'"
                ))
            res.register_next()
            self.next()
            return res.success(WhileNode(condition, body, True))
        return res.failure(InvalidSyntaxError(
            self.tok.pos_start, self.tok.pos_end,
            f"expected '{KEYWORDS['do']}' or new line"
        ))
    def for_expr(self):
        res = ParseResult()
        if not self.tok.matches(KEYWORD, KEYWORDS["for"]):
            return res.failure(InvalidSyntaxError(
                self.tok.pos_start, self.tok.pos_end,
                f"expected '{KEYWORDS['for']}'"
            ))
        res.register_next()
        self.next()
        if self.tok.type != IDENTIFIER:
            return res.failure(InvalidSyntaxError(
                self.tok.pos_start, self.tok.pos_end,
                f"expected identifier"
            ))
        var_name = self.tok
        res.register_next()
        self.next()
        if self.tok.type != EQ:
            return res.failure(InvalidSyntaxError(
                self.tok.pos_start, self.tok.pos_end,
                f"expected '='"
            ))
        res.register_next()
        self.next()
        start_value = res.register(self.expr())
        if res.error: return res
        if self.tok.type != SEP:
            return res.failure(InvalidSyntaxError(
                self.tok.pos_start, self.tok.pos_end,
                f"expected ','"
            ))
        res.register_next()
        self.next()
        end_value = res.register(self.expr())
        if res.error: return res
        if self.tok.matches(KEYWORD, KEYWORDS["step"]):
            res.register_next()
            self.next()
            step_value = res.register(self.expr())
            if res.error: return res
        else:
            step_value = None
        if self.tok.matches(KEYWORD, KEYWORDS["do"]):
            res.register_next()
            self.next()
            body = res.register(self.statement())
            if res.error: return res
            return res.success(ForNode(var_name, start_value, end_value, step_value, body))
        if self.tok.type == NL:
            res.register_next()
            self.next()
            body = res.register(self.statements())
            if res.error: return res
            if not self.tok.matches(KEYWORD, KEYWORDS["end"]):
                return res.failure(InvalidSyntaxError(
                    self.tok.pos_start, self.tok.pos_end,
                    f"expected '{KEYWORDS['end']}'"
                ))
            res.register_next()
            self.next()
            return res.success(ForNode(var_name, start_value, end_value, step_value, body, True))
        return res.failure(InvalidSyntaxError(
            self.tok.pos_start, self.tok.pos_end,
            f"expected '{KEYWORDS['do']}' or new line"
        ))
    def for_every_expr(self):
        res = ParseResult()
        if not self.tok.matches(KEYWORD, KEYWORDS["forEvery"]):
            return res.failure(InvalidSyntaxError(
                self.tok.pos_start, self.tok.pos_end,
                f"expected '{KEYWORDS['forEvery']}'"
            ))
        res.register_next()
        self.next()
        if self.tok.type != IDENTIFIER:
            return res.failure(InvalidSyntaxError(
                self.tok.pos_start, self.tok.pos_end,
                f"expected identifier"
            ))
        var_name = self.tok
        res.register_next()
        self.next()
        if not self.tok.matches(KEYWORD, KEYWORDS["of"]):
            return res.failure(InvalidSyntaxError(
                self.tok.pos_start, self.tok.pos_end,
                f"expected '{KEYWORDS['of']}'"
            ))
        res.register_next()
        self.next()
        list_value = res.register(self.expr())
        if res.error: return res
        if self.tok.matches(KEYWORD, KEYWORDS["do"]):
            res.register_next()
            self.next()
            body = res.register(self.statement())
            if res.error: return res
            return res.success(ForEveryNode(var_name, list_value, body))
        if self.tok.type == NL:
            res.register_next()
            self.next()
            body = res.register(self.statements())
            if res.error: return res
            if not self.tok.matches(KEYWORD, KEYWORDS["end"]):
                return res.failure(InvalidSyntaxError(
                    self.tok.pos_start, self.tok.pos_end,
                    f"expected '{KEYWORDS['end']}'"
                ))
            res.register_next()
            self.next()
            return res.success(ForEveryNode(var_name, list_value, body, True))
        return res.failure(InvalidSyntaxError(
            self.tok.pos_start, self.tok.pos_end,
            f"expected '{KEYWORDS['do']}' or new line"
        ))
    def if_expr(self):
        res = ParseResult()
        all_cases = res.register(self.if_expr_cases(KEYWORDS["if"]))
        if res.error: return res
        cases, else_case = all_cases
        return res.success(IfNode(cases, else_case))
    def elif_expr(self):
        return self.if_expr_cases(KEYWORDS["elif"])
    def else_expr(self):
        res = ParseResult()
        else_case = None
        if self.tok.matches(KEYWORD, KEYWORDS["else"]):
            res.register_next()
            self.next()
            if self.tok.type == NL:
                res.register_next()
                self.next()
                statements = res.register(self.statements())
                if res.error: return res
                else_case = (statements, True)
                if self.tok.matches(KEYWORD, KEYWORDS["end"]):
                    res.register_next()
                    self.next()
                else:
                    return res.failure(InvalidSyntaxError(
                        self.tok.pos_start, self.tok.pos_end,
                        f"expected {KEYWORDS['end']}"
                    ))
            else:
                statement = res.register(self.statement())
                if res.error: return res
                else_case = (statement, False)
        return res.success(else_case)
    def elif_or_else_expr(self):
        res = ParseResult()
        cases, else_case = [], None
        if self.tok.matches(KEYWORD, KEYWORDS["elif"]):
            all_cases = res.register(self.elif_expr())
            if res.error: return res
            cases, else_case = all_cases
        else:
            else_case = res.register(self.else_expr())
            if res.error: return res
        return res.success((cases, else_case))
    def if_expr_cases(self, kw):
        res = ParseResult()
        cases = []
        else_case = None
        if not self.tok.matches(KEYWORD, kw):
            return res.failure(InvalidSyntaxError(
                self.tok.pos_start, self.tok.pos_end,
                f"expected '{kw}'"
            ))
        res.register_next()
        self.next()
        condition = res.register(self.expr())
        if res.error: return res
        if self.tok.type == NL:
            res.register_next()
            self.next()
            statements = res.register(self.statements())
            if res.error: return res
            cases.append((condition, statements, True))
            if self.tok.matches(KEYWORD, KEYWORDS["end"]):
                res.register_next()
                self.next()
            else:
                all_cases = res.register(self.elif_or_else_expr())
                if res.error: return res
                new_cases, else_case = all_cases
                cases.extend(new_cases)
        else:
            res.register_next()
            self.next()
            statement = res.register(self.statement())
            if res.error: return res
            cases.append((condition, statement, False))
            all_cases = res.register(self.elif_or_else_expr())
            if res.error: return res
            new_cases, else_case = all_cases
            cases.extend(new_cases)
        return res.success((cases, else_case))
    def list_expr(self):
        res = ParseResult()
        element_nodes = []
        pos_start = self.tok.pos_start.copy()
        if self.tok.type != LISTIN:
            return res.failure(InvalidSyntaxError(
                self.tok.pos_start, self.tok.pos_end,
                f"expected '['"
            ))
        res.register_next()
        self.next()
        if self.tok.type == LISTOUT:
            res.register_next()
            self.next()
        else:
            element_nodes.append(res.register(self.expr()))
            if res.error: return res.failure(InvalidSyntaxError(
                self.tok.pos_start, self.tok.pos_end,
                f"expected ']', {INVALIDSYNTAX_ALL}"
            ))
            while self.tok.type == SEP:
                res.register_next()
                self.next()
                element_nodes.append(res.register(self.expr()))
                if res.error: return res
            if self.tok.type != LISTOUT:
                return res.failure(InvalidSyntaxError(
                    self.tok.pos_start, self.tok.pos_end,
                    f"expected ',', ']'"
                ))
            res.register_next()
            self.next()
        return res.success(ListNode(element_nodes, pos_start, self.tok.pos_end.copy()))
    def table_expr(self):
        res = ParseResult()
        key_nodes = []
        element_nodes = []
        pos_start = self.tok.pos_start.copy()
        if self.tok.type != TABLEIN:
            return res.failure(InvalidSyntaxError(
                self.tok.pos_start, self.tok.pos_end,
                "expected '{'"
            ))
        res.register_next()
        self.next()
        if self.tok.type == TABLEOUT:
            res.register_next()
            self.next()
        else:
            key = res.register(self.expr())
            if res.error: return res
            if self.tok.type != REP:
                return res.failure(InvalidSyntaxError(
                    self.tok.pos_start, self.tok.pos_end,
                    f"expected ':'"
                ))
            key_nodes.append(key)
            res.register_next()
            self.next()
            element_nodes.append(res.register(self.expr()))
            if res.error: return res
            while self.tok.type == SEP:
                res.register_next()
                self.next()
                key = res.register(self.expr())
                if res.error: return res
                if self.tok.type != REP:
                    return res.failure(InvalidSyntaxError(
                        self.tok.pos_start, self.tok.pos_end,
                        f"expected ':'"
                    ))
                key_nodes.append(key)
                res.register_next()
                self.next()
                element_nodes.append(res.register(self.expr()))
                if res.error: return res
            if self.tok.type != TABLEOUT:
                return res.failure(InvalidSyntaxError(
                    self.tok.pos_start, self.tok.pos_end,
                    "expected ',', '}'"
                ))
            res.register_next()
            self.next()
        return res.success(TableNode(element_nodes, key_nodes, pos_start, self.tok.pos_end.copy()))
    def atom(self):
        res = ParseResult()
        tok = self.tok
        if tok.type == NUMBER: # factor NumberNode
            res.register_next()
            self.next()
            return res.success(NumberNode(tok))
        if tok.type == NULL:
            res.register_next()
            self.next()
            return res.success(NullNode(tok))
        if tok.type in (TRUE, FALSE):
            res.register_next()
            self.next()
            return res.success(BoolNode(tok))
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
        elif tok.type == LISTIN: # factor EVALIN
            list_expr = res.register(self.list_expr())
            if res.error: return res
            return res.success(list_expr)
        elif tok.type == TABLEIN: # factor table in
            table_expr = res.register(self.table_expr())
            if res.error: return res
            return res.success(table_expr)
        elif tok.type == STRING:
            res.register_next()
            self.next()
            return res.success(StringNode(tok))
        elif tok.matches(KEYWORD, KEYWORDS["if"]):
            if_expr = res.register(self.if_expr())
            if res.error: return res
            return res.success(if_expr)
        elif tok.matches(KEYWORD, KEYWORDS["for"]):
            for_expr = res.register(self.for_expr())
            if res.error: return res
            return res.success(for_expr)
        elif tok.matches(KEYWORD, KEYWORDS["forEvery"]):
            for_every_expr = res.register(self.for_every_expr())
            if res.error: return res
            return res.success(for_every_expr)
        elif tok.matches(KEYWORD, KEYWORDS["while"]):
            while_expr = res.register(self.while_expr())
            if res.error: return res
            return res.success(while_expr)
        elif tok.matches(KEYWORD, KEYWORDS["function"]):
            func_def = res.register(self.func_def())
            if res.error: return res
            return res.success(func_def)
        return res.failure(InvalidSyntaxError(
            tok.pos_start, tok.pos_end,
            f"expected {INVALIDSYNTAX_ALL}"
        ))
    def call(self):
        res = ParseResult()
        atom = res.register(self.atom())
        if res.error: return res
        if self.tok.type == EVALIN:
            res.register_next()
            self.next()
            arg_nodes = []
            if self.tok.type == EVALOUT:
                res.register_next()
                self.next()
            else:
                arg_nodes.append(res.register(self.expr()))
                if res.error: return res.failure(InvalidSyntaxError(
                    self.tok.pos_start, self.tok.pos_end,
                    f"expected ')', {INVALIDSYNTAX_ALL}"
                ))
                while self.tok.type == SEP:
                    res.register_next()
                    self.next()
                    arg_nodes.append(res.register(self.expr()))
                    if res.error: return res
                if self.tok.type != EVALOUT:
                    return res.failure(InvalidSyntaxError(
                        self.tok.pos_start, self.tok.pos_end,
                        f"expected ',', ')'"
                    ))
                res.register_next()
                self.next()
            return res.success(CallNode(atom, arg_nodes))
        return res.success(atom)
    def index(self):
        return self.bin_op(self.call, [INDEX], self.power)
    def power(self):
        return self.bin_op(self.index, [POW, MOD], self.factor)
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
        return self.bin_op(self.factor, [MUL, DIV, DIVINT])
    def arith_expr(self):
        return self.bin_op(self.term, (PLUS, MINUS))
    def comp_expr(self):
        res = ParseResult()
        if self.tok.matches(KEYWORD, KEYWORDS["bool_not"]):
            op_tok = self.tok
            res.register_next()
            self.next()
            node = res.register(self.comp_expr())
            if res.error: return res
            return res.success(UnaryOpNode(op_tok, node))
        node = res.register(self.bin_op(self.arith_expr, [EE, NE, LT, GT, LTE, GTE, IN]))
        if res.error: return res.failure(InvalidSyntaxError(
            self.tok.pos_start, self.tok.pos_end,
            f"expected {INVALIDSYNTAX_ALL}"
        ))
        return res.success(node)
    def expr(self):
        res = ParseResult()
        if self.tok.matches(KEYWORD, KEYWORDS["use"]):
            res.register_next()
            self.next()
            file_name = res.register(self.expr())
            if res.error: return res
            return res.success(UseNode(file_name))
        if self.tok.type == IDENTIFIER:
            var_name = self.tok
            res.register_next()
            self.next()
            if self.tok.type == EQ:
                res.register_next()
                self.next()
                expr = res.register(self.expr())
                return res.success(VarAssignNode(var_name, expr))
            self.reverse()
        if self.tok.matches(INC):
            res.register_next()
            self.next()
            if self.tok.type != IDENTIFIER: return res.failure(InvalidSyntaxError(
                self.tok.pos_start, self.tok.pos_end,
                "expected identifier"
            ))
            var_name = self.tok
            res.register_next()
            self.next()
            if res.error: return res
            return res.success(VarIncrementationNode(var_name))
        node = res.register(self.bin_op(self.comp_expr, [(KEYWORD, KEYWORDS["bool_and"]), (KEYWORD, KEYWORDS["bool_or"])]))
        if res.error: return res.failure(InvalidSyntaxError(
                self.tok.pos_start, self.tok.pos_end,
                f"expected {INVALIDSYNTAX_START}"
            ))
        return res.success(node)
    def statement(self):
        res = ParseResult()
        pos_start = self.tok.pos_start.copy()
        if self.tok.matches(KEYWORD, KEYWORDS["return"]):
            res.register_next()
            self.next()
            expr = res.try_register(self.expr())
            if not expr:
                self.reverse(res.to_reverse_count)
            return res.success(ReturnNode(expr, pos_start, self.tok.pos_start.copy()))
        if self.tok.matches(KEYWORD, KEYWORDS["next"]):
            res.register_next()
            self.next()
            return res.success(NextNode(pos_start, self.tok.pos_start.copy()))
        if self.tok.matches(KEYWORD, KEYWORDS["break"]):
            res.register_next()
            self.next()
            return res.success(BreakNode(pos_start, self.tok.pos_start.copy()))
        expr = res.register(self.expr())
        if res.error: return res.failure(InvalidSyntaxError(
            pos_start, self.tok.pos_end.copy(),
            f"expected '{KEYWORDS['return']}', '{KEYWORDS['next']}', '{KEYWORDS['break']}', {INVALIDSYNTAX_START}"
        ))
        return res.success(expr)
    def statements(self):
        res = ParseResult()
        statements = []
        pos_start = self.tok.pos_start.copy()
        while self.tok.type == NL:
            res.register_next()
            self.next()
        statement = res.register(self.statement())
        if res.error: return res
        statements.append(statement)
        more_statements = True
        while True:
            nl_count = 0
            while self.tok.type == NL:
                res.register_next()
                self.next()
                nl_count += 1
            if nl_count == 0: more_statements = False
            if not more_statements: break
            statement = res.try_register(self.statement())
            if not statement:
                self.reverse(res.to_reverse_count)
                more_statements = False
                continue
            statements.append(statement)
        return res.success(ListNode(statements, pos_start, self.tok.pos_end.copy()))
    def bin_op(self, func1, ops: list, func2=None):
        if func2 is None:
            func2 = func1
        res = ParseResult()
        left = res.register(func1())
        if res.error: return res
        while self.tok.type in ops or (self.tok.type, self.tok.value) in ops:
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
        self.func_return_value = None
        self.loop_next = False
        self.loop_break = False
    def reset(self):
        self.value = None
        self.error = None
        self.func_return_value = None
        self.loop_next = False
        self.loop_break = False
    def register(self, res):
        if res.should_return(): self.error = res.error
        self.func_return_value = res.func_return_value
        self.loop_next = res.loop_next
        self.loop_break = res.loop_break
        return res.value
    def success(self, value):
        self.reset()
        self.value = value
        return self
    def success_return(self, value):
        self.reset()
        self.func_return_value = value
        return self
    def success_next(self):
        self.reset()
        self.loop_next = True
        return self
    def success_break(self):
        self.reset()
        self.loop_break = True
        return self
    def failure(self, error):
        self.reset()
        self.error = error
        return self
    def should_return(self):
        return self.error or self.func_return_value or self.loop_next or self.loop_break

"""VALUES"""
class Value:
    def __init__(self):
        self.set_pos()
        self.set_context()
    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self
    def set_context(self, context=None):
        self.context = context
        return self
    def comp_both(self, other):
        error = None
        if isinstance(self, (Number, Bool, String)):
            left = self
        else:
            left, error = self.as_number()
            if error: return None, None, error
        if isinstance(other, (Number, Bool, String)):
            right = other
        else:
            right, error = other.as_number()
            if error: return None, None, error
        return left, right, error
    def bool_both(self, other):
        right, error = other.as_bool()
        if error: return None, None, error
        left, error = self.as_bool()
        if error: return None, None, error
        return left, right, error
    def number_both(self, other):
        right, error = other.as_number()
        if error: return None, None, error
        left, error = self.as_number()
        if error: return None, None, error
        return left, right, error
    def add(self, other):
        if isinstance(self, List):
            elements = self.elements
            elements.append(other)
            return List(elements), None
        if isinstance(self, String):
            if isinstance(other, String):
                return String(self.value + other.value), None
            return None, self.illagel_operation(other)
        left, right, error = self.number_both(other)
        if error: return None, error
        return Number(left.value + right.value), None
    def sub(self, other):
        if isinstance(self, List):
            right, error = other.as_number()
            if error: return None, error
            elements = self.elements
            elements.pop(right.value)
            return List(elements), None
        left, right, error = self.number_both(other)
        if error: return None, error
        return Number(left.value - right.value), None
    def mul(self, other):
        if isinstance(self, List) and isinstance(other, List):
            elements = self.elements
            elements.extend(other.elements)
            return List(elements), None
        left, right, error = self.number_both(other)
        if error: return None, error
        return Number(left.value * right.value), None
    def div(self, other):
        left, right, error = self.number_both(other)
        if error: return None, error
        if right.value == 0: return None, RTError(self.pos_start, other.pos_end, "division by zero", self.context)
        return Number(left.value / right.value), None
    def divint(self, other):
        left, right, error = self.number_both(other)
        if error: return None, error
        if right.value == 0: return None, RTError(self.pos_start, other.pos_end, "division by zero", self.context)
        return Number(left.value // right.value), None
    def pow(self, other):
        left, right, error = self.number_both(other)
        if error: return None, error
        return Number(left.value ** right.value), None
    def index(self, other):
        if isinstance(self, List):
            right, error = other.as_number()
            try:
                return self.elements[right.value], None
            except IndexError:
                return None, RTError(
                    other.pos_start, other.pos_end,
                    f"index out of range ({right.value} is over {len(self.elements) - 1})", self.context
                )
        if isinstance(self, String):
            right, error = other.as_number()
            try:
                return String(self.value[right.value]), None
            except IndexError:
                return None, RTError(
                    other.pos_start, other.pos_end,
                    f"index out of range ({right.value} is over {len(self.value) - 1})", self.context
                )
        if isinstance(self, Table):
            right, error = other.as_string()
            try:
                return self.table[right.value], None
            except KeyError:
                return None, RTError(
                    other.pos_start, other.pos_end,
                    f"key not found", self.context
                )
        return None, self.illagel_operation(other)
    def neg(self):
        left, error = self.as_number()
        if error: return None, error
        return Number(-left.value), None
    def _not(self):
        left, error = self.as_bool()
        if error: return None, error
        return Bool(not left.value), None
    def _and(self, other):
        left, right, error = self.bool_both(other)
        if error: return None, error
        return Bool(left.value and right.value), None
    def _or(self, other):
        left, right, error = self.bool_both(other)
        if error: return None, error
        return Bool(left.value or right.value), None
    def _in(self, other):
        if isinstance(other, List):
            values = []
            for val in other.elements:
                if isinstance(val, List):
                    values.append(val.elements)
                else:
                    values.append(val.value)
            if isinstance(self, List):
                return Bool(self.elements in values), None
            else:
                return Bool(self.value in values), None
        if isinstance(other, String) and isinstance(self, String):
            return Bool(self.value in other.value), None
        if isinstance(other, Table) and isinstance(self, String):
            return Bool(self.value in other.value), None
        return None, self.illagel_operation(other)
    def ee(self, other):
        left, right, error = self.comp_both(other)
        if error: return None, error
        return Bool(left.value == right.value), None
    def ne(self, other):
        left, right, error = self.comp_both(other)
        if error: return None, error
        return Bool(left.value != right.value), None
    def lt(self, other):
        left, right, error = self.comp_both(other)
        if error: return None, error
        return Bool(left.value < right.value), None
    def gt(self, other):
        left, right, error = self.comp_both(other)
        if error: return None, error
        return Bool(left.value > right.value), None
    def lte(self, other):
        left, right, error = self.comp_both(other)
        if error: return None, error
        return Bool(left.value <= right.value), None
    def gte(self, other):
        left, right, error = self.comp_both(other)
        if error: return None, error
        return Bool(left.value >= right.value), None
    def as_number(self):
        return None, self.illagel_cast("number")
    def as_string(self):
        return None, self.illagel_cast("string")
    def as_bool(self):
        return None, self.illagel_cast("bool")
    def to_num(self):
        return None, self.illagel_cast("number")
    def to_bool(self):
        return None, self.illagel_cast("bool")
    def to_str(self):
        return None, self.illagel_cast("string")
    def to_list(self):
        return None, self.illagel_cast("list")
    def is_true(self):
        return False
    def illagel_cast(self, cast_to):
        return RTError(
            self.pos_start, self.pos_end,
            "illegal cast to "+str(cast_to), self.context
        )
    def illagel_operation(self, other=None):
        if not other: other = self
        return RTError(
            self.pos_start, self.pos_end,
            "illegal operation", self.context
        )
    def call(self, args):
        return None, self.illagel_operation()
    def __repr__(self):
        return "?"
class Null(Value):
    def copy(self):
        copy = Null()
        copy.set_context(self.context)
        copy.set_pos(self.pos_start, self.pos_end)
        return copy
    def as_number(self):
        return Number(0), None
    def as_bool(self):
        return Bool(False), None
    def as_string(self):
        return String("null"), None
    def to_num(self):
        return Number(0), None
    def to_bool(self):
        return Bool(False), None
    def to_str(self):
        return String("null"), None
    def is_true(self):
        return False
    def __repr__(self):
        return "null"
class Number(Value):
    def __init__(self, value):
        super().__init__()
        if float(value) == int(value):
            self.value = int(value)
        else:
            self.value = float(value)
    def copy(self):
        copy = Number(self.value)
        copy.set_context(self.context)
        copy.set_pos(self.pos_start, self.pos_end)
        return copy
    def as_number(self):
        return self.copy(), None
    def as_string(self):
        return String(str(self.value)), None
    def as_bool(self):
        return Bool(self.value != 0), None
    def to_num(self):
        return Number(self.value), None
    def to_bool(self):
        return Bool(self.value != 0), None
    def to_str(self):
        return String(str(self.value)), None
    def is_true(self):
        return self.value != 0
    def __repr__(self):
        return f"{self.value}"
class Bool(Value):
    def __init__(self, value):
        super().__init__()
        self.value = value
    def __repr__(self):
        return str(self.value).lower()
    def copy(self):
        copy = Bool(self.value)
        copy.set_context(self.context)
        copy.set_pos(self.pos_start, self.pos_end)
        return copy
    def as_number(self):
        return Number(1 if self.value else 0), None
    def as_string(self):
        return String(str(self.value).lower()), None
    def as_bool(self):
        return Bool(self.value), None
    def to_num(self):
        return Number(1 if self.value else 0), None
    def to_bool(self):
        return Bool(self.value), None
    def to_str(self):
        return String(str(self.value).lower()), None
    def is_true(self):
        return self.value
class String(Value):
    def __init__(self, value):
        super().__init__()
        self.value = str(value)
    def copy(self):
        copy = String(self.value)
        copy.set_context(self.context)
        copy.set_pos(self.pos_start, self.pos_end)
        return copy
    def as_string(self):
        return self.copy(), None
    def to_num(self):
        try:
            return Number(int(self.value)), None
        except ValueError:
            try:
                return Number(float(self.value)), None
            except ValueError:
                return None, self.illagel_cast("number")
    def to_bool(self):
        if self.value == "true":
            return Bool(True), None
        if self.value == "false":
            return Bool(False), None
        return None, self.illagel_cast("bool")
    def to_str(self):
        return String(self.value), None
    def to_list(self):
        return List([String(x) for x in list(self.value)]), None
    def __str__(self):
        return self.value
    def is_true(self):
        return len(self.value) > 0
    def __repr__(self):
        return f'"{self.value}"'
class List(Value):
    def __init__(self, elements):
        super().__init__()
        self.elements = elements
    def copy(self):
        copy = List(self.elements)
        copy.set_context(self.context)
        copy.set_pos(self.pos_start, self.pos_end)
        return copy
    def to_list(self):
        return List(self.elements), None
    def __repr__(self):
        return f"[{', '.join([repr(x) for x in self.elements])}]"
class Table(Value):
    def __init__(self, table: dict):
        super().__init__()
        self.table = table
    def copy(self):
        copy = Table(self.table)
        copy.set_context(self.context)
        copy.set_pos(self.pos_start, self.pos_end)
        return copy
    def __repr__(self):
        return "{" + f"{', '.join([f'{repr(x)}: {repr(self.table[x])}' for x in self.table])}" + "}"
class BaseFunction(Value):
    def __init__(self, name):
        super().__init__()
        self.name = name or "<anonymous>"
    def copy(self):
        copy = BuiltInFunction(self.name)
        copy.set_context(self.context)
        copy.set_pos(self.pos_start, self.pos_end)
        return copy
    def generate_new_context(self):
        new_context = Context(self.name, self.context, self.pos_start)
        new_context.vars = Vars(new_context.parent.vars)
        return new_context
    def check_args(self, arg_names, args):
        res = RTResult()
        if len(args) > len(arg_names):
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                f"{len(args) - len(arg_names)} too many args passed into '{self.name}'",
                self.context
            ))
        if len(args) < len(arg_names):
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                f"{len(arg_names) - len(args)} too few args passed into '{self.name}'",
                self.context
            ))
        return res.success(None)
    def plot_args(self, arg_names, args, exec_ctx):
        for i in range(len(args)):
            arg_name = arg_names[i]
            arg_value = args[i]
            arg_value.set_context(exec_ctx)
            exec_ctx.vars.set(arg_name, arg_value)
    def check_and_plot_args(self, arg_names, args, exec_ctx):
        res = RTResult()
        res.register(self.check_args(arg_names, args))
        if res.should_return(): return res
        self.plot_args(arg_names, args, exec_ctx)
        return res.success(None)
class Function(BaseFunction):
    def __init__(self, name, body_node, arg_names, auto_return):
        super().__init__(name)
        self.body_node = body_node
        self.arg_names = arg_names
        self.auto_return = auto_return
    def copy(self):
        copy = Function(self.name, self.body_node, self.arg_names, self.auto_return)
        copy.set_context(self.context)
        copy.set_pos(self.pos_start, self.pos_end)
        return copy
    def call(self, args):
        res = RTResult()
        interpreter = Interpreter()
        new_context = self.generate_new_context()
        res.register(self.check_and_plot_args(self.arg_names, args, new_context))
        if res.should_return(): return res
        value = res.register(interpreter.visit(self.body_node, new_context))
        if res.should_return() and res.func_return_value is None: return res
        return res.success((value if self.auto_return else None) or res.func_return_value or Null())
    def __repr__(self):
        return f"<function '{self.name}'>"
class BuiltInFunction(BaseFunction):
    def __init__(self, name):
        super().__init__(name)
    def copy(self):
        copy = BuiltInFunction(self.name)
        copy.set_context(self.context)
        copy.set_pos(self.pos_start, self.pos_end)
        return copy
    def call(self, args):
        res = RTResult()
        exec_ctx = self.generate_new_context()
        method_name = f"execute_{self.name}"
        method = getattr(self, method_name, self.no_visit_method)
        res.register(self.check_and_plot_args(method.arg_names, args, exec_ctx))
        if res.should_return(): return res
        return_value = res.register(method(exec_ctx))
        if res.should_return(): return res
        return res.success(return_value)
    def no_visit_method(self, node, context):
        raise Exception(f"No execute_{self.name} method defined")
    def __repr__(self):
        return f"<built-in function {self.name}>"
    def execute_print(self, exec_ctx):
        print(str(exec_ctx.vars.get("print_value")))
        return RTResult().success(BuiltInFunction.print)
    execute_print.arg_names = ["print_value"]
    def execute_input(self, exec_ctx):
        return RTResult().success(String(input()))
    execute_input.arg_names = []
    def execute_input_num(self, exec_ctx):
        while True:
            text = input()
            try:
                number = int(text)
                break
            except ValueError:
                try:
                    number = float(text)
                    break
                except ValueError:
                    print(f"'{text}' is not a number")
        return RTResult().success(Number(number))
    execute_input_num.arg_names = []
    def execute_is_num(self, exec_ctx):
        return RTResult().success(Bool(isinstance(exec_ctx.vars.get("is_value"), Number)))
    execute_is_num.arg_names = ["is_value"]
    def execute_is_bool(self, exec_ctx):
        return RTResult().success(Bool(isinstance(exec_ctx.vars.get("is_value"), Bool)))
    execute_is_bool.arg_names = ["is_value"]
    def execute_is_str(self, exec_ctx):
        return RTResult().success(Bool(isinstance(exec_ctx.vars.get("is_value"), String)))
    execute_is_str.arg_names = ["is_value"]
    def execute_is_list(self, exec_ctx):
        return RTResult().success(Bool(isinstance(exec_ctx.vars.get("is_value"), List)))
    execute_is_list.arg_names = ["is_value"]
    def execute_is_func(self, exec_ctx):
        return RTResult().success(Bool(isinstance(exec_ctx.vars.get("is_value"), BaseFunction)))
    execute_is_func.arg_names = ["is_value"]
    def execute_to_num(self, exec_ctx):
        value = exec_ctx.vars.get("to_value")
        value, error = value.to_num()
        if error: return RTResult().failure(error)
        return RTResult().success(value)
    execute_to_num.arg_names = ["to_value"]
    def execute_to_bool(self, exec_ctx):
        value = exec_ctx.vars.get("to_value")
        value, error = value.to_bool()
        if error: return RTResult().failure(error)
        return RTResult().success(value)
    execute_to_bool.arg_names = ["to_value"]
    def execute_to_str(self, exec_ctx):
        value = exec_ctx.vars.get("to_value")
        value, error = value.to_str()
        if error: return RTResult().failure(error)
        return RTResult().success(value)
    execute_to_str.arg_names = ["to_value"]
    def execute_to_list(self, exec_ctx):
        value = exec_ctx.vars.get("to_value")
        value, error = value.to_list()
        if error: return RTResult().failure(error)
        return RTResult().success(value)
    execute_to_list.arg_names = ["to_value"]
    def execute_type(self, exec_ctx):
        value = exec_ctx.vars.get("type_value")
        return RTResult().success(String(type(value).__name__.lower()))
    execute_type.arg_names = ["type_value"]
    def execute_len(self, exec_ctx):
        value = exec_ctx.vars.get("len_value")
        if isinstance(value, List):
            return RTResult().success(Number(len(value.elements)))
        if isinstance(value, String):
            return RTResult().success(Number(len(value.value)))
        return RTResult().failure(RTError(self.pos_start, self.pos_end, "can only get length of list or string", exec_ctx))
    execute_len.arg_names = ["len_value"]
    def execute_run(self, exec_ctx):
        fn = exec_ctx.vars.get("run_file_name")
        if not isinstance(fn, String):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "argument must be a string", exec_ctx
            ))
        fn = fn.value
        try:
            with open(fn, "r") as f:
                script = f.read()
        except Exception as e:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                f"failed to load script '{fn}'\n{e}", exec_ctx
            ))
        _, error = run(fn, script)
        if error:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                f"failed to finish running script '{fn}'\n{error.as_string()}", exec_ctx
            ))
        return RTResult().success(BuiltInFunction.run)
    execute_run.arg_names = ["run_file_name"]
    def execute_sleep(self, exec_ctx):
        n = exec_ctx.vars.get("sleep_value")
        if not isinstance(n, Number):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "argument must be a number", exec_ctx
            ))
        sleep(n.value)
        return RTResult().success(BuiltInFunction.sleep)
    execute_sleep.arg_names = ["sleep_value"]
    def execute_time(self, exec_ctxs):
        return RTResult().success(Number(time()))
    execute_time.arg_names = []
    def execute_exit(self, exec_ctx):
        value = exec_ctx.vars.get("exit_value")
        if isinstance(value, Number) or isinstance(value, String):
            exit(value.value)
    execute_exit.arg_names = ["exit_value"]
    def execute_error(self, exec_ctx):
        head = exec_ctx.vars.get("error_head")
        value = exec_ctx.vars.get("error_value")
        if isinstance(value, String):
            return RTResult().failure(UserDefinedError(head.value, value.value))
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "argument has to be a string", exec_ctx
        ))
    execute_error.arg_names = ["error_head", "error_value"]
    def execute_set(self, exec_ctx):
        table = exec_ctx.vars.get("set_table")
        if not isinstance(table, Table): return RTResult().failure(RTError(
            table.pos_start, table.pos_end, "expected table", exec_ctx
        ))
        key = exec_ctx.vars.get("set_key")
        if not isinstance(key, String): return RTResult().failure(RTError(
            key.pos_start, key.pos_end, "expected string", exec_ctx
        ))
        value = exec_ctx.vars.get("set_value")
        table.table[key.value] = value
        exec_ctx.vars.set("set_table", table)
        return RTResult().success(Table(table.table))
    execute_set.arg_names = ["set_table", "set_key", "set_value"]
Number.pi                   = Number(pi)
String.empty                = String("")
List.empty                  = List([])
BuiltInFunction.print       = BuiltInFunction("print")
BuiltInFunction.input       = BuiltInFunction("input")
BuiltInFunction.input_num   = BuiltInFunction("input_num")
BuiltInFunction.is_num      = BuiltInFunction("is_num")
BuiltInFunction.is_bool     = BuiltInFunction("is_bool")
BuiltInFunction.is_str      = BuiltInFunction("is_str")
BuiltInFunction.to_num      = BuiltInFunction("to_num")
BuiltInFunction.to_bool     = BuiltInFunction("to_bool")
BuiltInFunction.to_str      = BuiltInFunction("to_str")
BuiltInFunction.to_list     = BuiltInFunction("to_list")
BuiltInFunction.is_list     = BuiltInFunction("is_list")
BuiltInFunction.is_func     = BuiltInFunction("is_func")
BuiltInFunction.type        = BuiltInFunction("type")
BuiltInFunction.len         = BuiltInFunction("len")
BuiltInFunction.run         = BuiltInFunction("run")
BuiltInFunction.sleep       = BuiltInFunction("sleep")
BuiltInFunction.time        = BuiltInFunction("time")
BuiltInFunction.exit        = BuiltInFunction("exit")
BuiltInFunction.error       = BuiltInFunction("error")
BuiltInFunction.set         = BuiltInFunction("set")

"""CONTEXT"""
class Context:
    def __init__(self, display_name, parent=None, parent_entry_pos=None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.vars = None

"""SYMBOL TABLE"""
class Vars:
    def __init__(self, parent=None):
        self.vars = {}
        self.parent = parent
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
    def visit(self, node, context: Context):
        method_name = f"visit_{type(node).__name__}"
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)
    def no_visit_method(self, node, context: Context):
        raise Exception(f"no visit_{type(node).__name__} method defined")
    def visit_NumberNode(self, node: NumberNode, context: Context): # number
        return RTResult().success(Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end))
    def visit_StringNode(self, node: StringNode, context: Context):
        return RTResult().success(String(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end))
    def visit_ListNode(self, node: ListNode, context: Context):
        res = RTResult()
        elements = []
        for element_node in node.element_nodes:
            elements.append(res.register(self.visit(element_node, context)))
            if res.should_return(): return res
        return res.success(List(elements).set_context(context).set_pos(node.pos_start, node.pos_end))
    def visit_TableNode(self, node: TableNode, context: Context):
        res = RTResult()
        table = {}
        for i in range(len(node.element_nodes)):
            key = res.register(self.visit(node.key_nodes[i], context))
            if res.should_return(): return res
            if not isinstance(key, String): return res.failure(RTError(
                key.pos_start, key.pos_end, "expected a string", context
            ))
            value = res.register(self.visit(node.element_nodes[i], context))
            if res.should_return(): return res
            table[key.value] = value
        return res.success(Table(table).set_context(context).set_pos(node.pos_start, node.pos_end))
    def visit_NullNode(self, node: NullNode, context: Context):
        return RTResult().success(Null().set_context(context).set_pos(node.pos_start, node.pos_end))
    def visit_BoolNode(self, node: NullNode, context: Context):
        return RTResult().success(Bool(node.tok.type == TRUE).set_context(context).set_pos(node.pos_start, node.pos_end))
    def visit_VarAccessNode(self, node: VarAccessNode, context: Context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = context.vars.get(var_name)
        if not value:
            return res.failure(RTError(
                node.pos_start, node.pos_end,
                f"'{var_name}' is not defined" if not var_name in ["father", "dad", "life"] else f"you have no {var_name}",
                context
            ))
        value = value.copy().set_pos(node.pos_start, node.pos_end).set_context(context)
        return res.success(value)
    def visit_VarAssignNode(self, node: VarAssignNode, context: Context):
        res = RTResult()
        var_name = node.var_name_tok.value
        if node.value_node:
            value = res.register(self.visit(node.value_node, context))
            if res.should_return(): return res
            context.vars.set(var_name, value)
            return res.success(value)
        else:
            context.vars.set(var_name, Null())
            return res.success(Null())
    def visit_BinOpNode(self, node: BinOpNode, context: Context): # bin op
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.should_return(): return res
        right = res.register(self.visit(node.right_node, context))
        if res.should_return(): return res
        result = None
        error = None
        if node.op_tok.type == PLUS: # add
            result, error = left.add(right)
        if node.op_tok.type == MINUS: # sub
            result, error = left.sub(right)
        if node.op_tok.type == MUL: # mul
            result, error = left.mul(right)
        if node.op_tok.type == DIV: # div
            result, error = left.div(right)
        if node.op_tok.type == DIVINT: # div
            result, error = left.divint(right)
        if node.op_tok.type == POW: # pow
            result, error = left.pow(right)
        if node.op_tok.type == EE: # ee
            result, error = left.ee(right)
        if node.op_tok.type == NE: # ne
            result, error = left.ne(right)
        if node.op_tok.type == LT: # lt
            result, error = left.lt(right)
        if node.op_tok.type == GT: # gt
            result, error = left.gt(right)
        if node.op_tok.type == LTE: # lte
            result, error = left.lte(right)
        if node.op_tok.type == GTE: # gte
            result, error = left.gte(right)
        if node.op_tok.type == IN: # gte
            result, error = left._in(right)
        if node.op_tok.type == INDEX: # gte
            result, error = left.index(right)
        if node.op_tok.type == MOD: # gte
            result, error = left.mod(right)
        if node.op_tok.matches(KEYWORD, KEYWORDS["bool_and"]): # and
            result, error = left._and(right)
        if node.op_tok.matches(KEYWORD, KEYWORDS["bool_or"]): # or
            result, error = left._or(right)
        if error: return res.failure(error)
        return res.success(result.set_pos(node.pos_start, node.pos_end))
    def visit_UnaryOpNode(self, node: UnaryOpNode, context: Context): # unary op
        res = RTResult()
        number = res.register(self.visit(node.node, context))
        if res.should_return(): return res
        error = None
        if node.op_tok.type == MINUS:
            number, error = number.neg()
        if node.op_tok.matches(KEYWORD, KEYWORDS["bool_not"]):
            number, error = number._not()
        if error: return res.failure(error)
        return res.success(number.set_pos(node.pos_start, node.pos_end))
    def visit_VarIncrementationNode(self, node: VarIncrementationNode, context: Context):
        res = RTResult()
        var_name = node.var_name_tok.value
        if res.should_return(): return res
        context.vars.set(var_name, Number(context.vars.get(var_name).value + 1))
        return res.success(context.vars.get(var_name))
    def visit_IfNode(self, node: IfNode, context: Context):
        res = RTResult()
        for condition, expr, return_null in node.cases:
            condintion_value = res.register(self.visit(condition, context))
            if res.should_return(): return res
            if condintion_value.is_true():
                expr_value = res.register(self.visit(expr, context))
                if res.should_return(): return res
                return res.success(Null() if return_null else expr_value)
        if node.else_case:
            expr, return_null = node.else_case
            else_value = res.register(self.visit(expr, context))
            if res.should_return(): return res
            return res.success(Null() if return_null else else_value)
        return res.success(Null())
    def visit_ForNode(self, node: ForNode, context: Context):
        res = RTResult()
        elements = []
        start_value = res.register(self.visit(node.start_value_node, context))
        if res.should_return(): return res
        end_value = res.register(self.visit(node.end_value_node, context))
        if res.should_return(): return res
        if node.step_value_node:
            step_value = res.register(self.visit(node.step_value_node, context))
        else:
            step_value = Number(1)
        if res.should_return(): return res
        i = start_value.value
        if step_value.value >= 0:
            condition = lambda: i < end_value.value
        else:
            condition = lambda: i > end_value.value
        while condition():
            context.vars.set(node.var_name_tok.value, Number(i))
            i += step_value.value
            value = res.register(self.visit(node.body_node, context))
            if res.should_return() and (not res.loop_next) and (not res.loop_break): return res
            if res.loop_next: continue
            if res.loop_break: break
            if value: elements.append(value)
        return res.success(Null() if node.return_null else List(elements).set_context(context).set_pos(node.pos_start, node.pos_end))
    def visit_ForEveryNode(self, node: ForEveryNode, context: Context):
        res = RTResult()
        elements = []
        list_value = res.register(self.visit(node.list_name_tok, context))
        for var_value in list_value.elements:
            context.vars.set(node.var_name_tok.value, var_value)
            value = res.register(self.visit(node.body_node, context))
            if res.should_return() and (not res.loop_next) and (not res.loop_break): return res
            if res.loop_next: continue
            if res.loop_break: break
            if value: elements.append(value)
        return res.success(Null() if node.return_null else List(elements).set_context(context).set_pos(node.pos_start, node.pos_end))
    def visit_WhileNode(self, node: WhileNode, context: Context):
        res = RTResult()
        elements = []
        while True:
            condition = res.register(self.visit(node.condition_node, context))
            if res.should_return(): return res
            if not condition.is_true(): break
            value = res.register(self.visit(node.body_node, context))
            if res.should_return() and (not res.loop_next) and (not res.loop_break): return res
            if res.loop_next: continue
            if res.loop_break: break
            if value: elements.append(value)
        return res.success(Null() if node.return_null else List(elements).set_context(context).set_pos(node.pos_start, node.pos_end))
    def visit_FuncDefNode(self, node: FuncDefNode, context: Context):
        res = RTResult()
        func_name = node.var_name_tok.value if node.var_name_tok else None
        body_node = node.body_node
        arg_names = [arg_name.value for arg_name in node.arg_name_toks]
        func_value = Function(func_name, body_node, arg_names, node.auto_return).set_context(context).set_pos(node.pos_start, node.pos_end)
        if node.var_name_tok:
            context.vars.set(func_name, func_value)
        return res.success(func_value)
    def visit_CallNode(self, node: CallNode, context: Context):
        res = RTResult()
        args = []
        value_to_call = res.register(self.visit(node.node_to_call, context))
        if res.should_return(): return res
        if not isinstance(value_to_call, BaseFunction): return res.failure(RTError(
            node.pos_start, node.pos_end, f"cannot call a non-function value", context
        ))
        value_to_call = value_to_call.copy().set_pos(node.pos_start, node.pos_end)
        for arg_node in node.arg_nodes:
            args.append(res.register(self.visit(arg_node, context)))
            if res.should_return(): return res
        return_value = res.register(value_to_call.call(args))
        if res.should_return(): return res
        return_value = return_value.copy().set_pos(node.pos_start, node.pos_end).set_context(context)
        return res.success(return_value)
    def visit_ReturnNode(self, node: ReturnNode, context: Context):
        res = RTResult()
        if node.node_to_return:
            value = res.register(self.visit(node.node_to_return, context))
            if res.should_return(): return res
        else:
            value = Null()
        return res.success_return(value)
    def visit_NextNode(self, node: NextNode, context: Context):
        return RTResult().success_next()
    def visit_BreakNode(self, node: BreakNode, context: Context):
        return RTResult().success_break()
    def visit_UseNode(self, node: UseNode, context: Context):
        res = RTResult()
        fn = res.register(self.visit(node.file_name_tok, context))
        file_name = ""
        if res.error: return res
        if not isinstance(fn, String):
            return res.failure(RTError(node.pos_start, node.pos_end, "expected string", context))
        try:
            with open(fn.value+".b", "r") as f:
                text = f.read()
                file_name = fn.value+".b"
        except FileNotFoundError:
            try:
                with open(fn.value + ".py", "r") as f:
                    text = f.read()
                    file_name = fn.value + ".py"
            except FileNotFoundError:
                return res.failure(RTError(node.pos_start, node.pos_end, f"use file '{file_name}' not found", context))
        if file_name.endswith(".py"):
            exec(text)
            return res.success(file_name)
        lexer = Lexer(file_name, text)
        tokens, error = lexer.make_tokens()
        if error: return res.failure(error)
        parser = Parser(tokens)
        ast = parser.parse()
        if ast.error: return res.failure(ast.error)
        for n in ast.node.element_nodes:
            if isinstance(n, FuncDefNode):
                res.register(self.visit_FuncDefNode(n, context))
                if res.error: return res
            if isinstance(n, VarAssignNode):
                res.register(self.visit_VarAssignNode(n, context))
                if res.error: return res
        return res.success(file_name)
"""RUN"""
global_vars = Vars()
global_vars.set("print", BuiltInFunction.print)
global_vars.set("input", BuiltInFunction.input)
global_vars.set("inputNum", BuiltInFunction.input_num)
global_vars.set("isNum", BuiltInFunction.is_num)
global_vars.set("isBool", BuiltInFunction.is_bool)
global_vars.set("isStr", BuiltInFunction.is_str)
global_vars.set("isList", BuiltInFunction.is_list)
global_vars.set("toNum", BuiltInFunction.to_num)
global_vars.set("toBool", BuiltInFunction.to_bool)
global_vars.set("toStr", BuiltInFunction.to_str)
global_vars.set("toList", BuiltInFunction.to_list)
global_vars.set("isFunc", BuiltInFunction.is_func)
global_vars.set("type", BuiltInFunction.type)
global_vars.set("len", BuiltInFunction.len)
global_vars.set("run", BuiltInFunction.run)
global_vars.set("sleep", BuiltInFunction.sleep)
global_vars.set("time", BuiltInFunction.time)
global_vars.set("exit", BuiltInFunction.exit)
global_vars.set("error", BuiltInFunction.error)
global_vars.set("set", BuiltInFunction.set)
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
    fn = argv[1]
    try:
        with open(fn, "r") as f:
            script = f.read()
    except Exception as e:
        print(f"failed to load script '{fn}'\n{e}")
    _, error = run(fn, script)
    if error:
        print(f"failed to finish running script '{fn}'\n{error.as_string()}")
