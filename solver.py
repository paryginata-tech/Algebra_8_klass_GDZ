# -*- coding: utf-8 -*-
"""
Модуль пошагового решения для офлайн-решебника.
Работает внутри Pyodide (sympy в браузере).

Поддерживает 4 типа заданий:
  - reduce:    сократить дробь num/den
  - addsub:    сложить/вычесть 2-3 дроби с разными знаменателями
  - muldiv:    умножить/разделить 2 дроби
  - equation:  решить уравнение (линейное/квадратное/дробно-рациональное)

Все функции возвращают dict с ключами: steps (list[str]), odz (str|None), answer (str).

ВАЖНО (уроки прототипа, см. чат): sympy АВТОМАТИЧЕСКИ распределяет деление на
число/выражение внутри суммы прямо на уровне арифметики: (y-4)/3 превращается
в y/3 - 4/3 ещё до печати. Поэтому финальный ответ нигде не строится как
"sympy-объект / sympy-объект" напрямую — только через _frac_str(), которая
работает со строками, а сокращение дробей всегда идёт через списки множителей
(factor_list), а не через expand()+cancel(), иначе теряется факторизованный вид.
"""
import sympy as sp
import json

_SYMS = sp.symbols('x y z a b m n p q c d k', real=True)
_SYMTAB = {str(s): s for s in _SYMS}

_TRANSFORMS = sp.parsing.sympy_parser.standard_transformations + (
    sp.parsing.sympy_parser.implicit_multiplication_application,
    sp.parsing.sympy_parser.convert_xor,
)

def _parse(s):
    s = s.strip()
    if not s:
        raise ValueError("пустое выражение")
    expr = sp.parsing.sympy_parser.parse_expr(s, local_dict=_SYMTAB, transformations=_TRANSFORMS)
    # Десятичные литералы (0.5 и т.п.) парсятся как sympy Float, а Float
    # "заражает" плавающей точкой всю дальнейшую арифметику (7.0, -1.00000000000000
    # вместо точных 7 и -1). Сразу переводим их в точные дроби (0.5 -> 1/2).
    floats = expr.atoms(sp.Float)
    if floats:
        expr = expr.subs({f: sp.nsimplify(f, rational=True) for f in floats})
    return expr

def _fmt(expr):
    """Человекочитаемая печать: ** -> ^, sqrt(n) -> √n."""
    txt = sp.sstr(expr, order=None)
    txt = txt.replace('**', '^')
    import re
    txt = re.sub(r'sqrt\(([^()]+)\)', r'√\1', txt)
    return txt

def _needs_parens(expr):
    """Нужны ли скобки при подстановке expr как части дроби num/den."""
    if expr.is_Atom:
        return False
    if expr.is_Mul or expr.is_Pow:
        return False  # a*b, a^n сами по себе не нуждаются в скобках в числителе/знаменателе дроби
    return True  # Add и прочее — да

def _frac_str(num, den):
    """Строит ТЕКСТ дроби num/den вручную (не доверяя sympy-делению!)."""
    if den == 1:
        return _fmt(num)
    num_s = f"({_fmt(num)})" if _needs_parens(num) else _fmt(num)
    den_s = f"({_fmt(den)})" if (den.is_Add or den.is_Mul) else _fmt(den)
    return f"{num_s}/{den_s}"


def _factor_dict(expr):
    """factor_list -> dict{factor: power}, плюс числовой коэффициент отдельно."""
    coeff, flist = sp.factor_list(expr)
    d = {}
    for f, p in flist:
        d[f] = d.get(f, 0) + p
    return coeff, d

def _cancel_via_factors(num_expr, den_expr):
    """Сокращает num_expr/den_expr, работая СПИСКАМИ МНОЖИТЕЛЕЙ (а не expand),
    чтобы результат остался в красивом факторизованном виде, а не превратился
    в раскрытый многочлен. Возвращает (result_num, result_den, common_factors_str)."""
    ncoeff, nd = _factor_dict(num_expr)
    dcoeff, dd = _factor_dict(den_expr)

    common = []
    for f in list(nd.keys()):
        if f in dd:
            p = min(nd[f], dd[f])
            if p > 0:
                common.append((f, p))
                nd[f] -= p
                dd[f] -= p
                if nd[f] == 0: del nd[f]
                if dd[f] == 0: del dd[f]

    num_coeff_reduced = sp.nsimplify(ncoeff)
    den_coeff_reduced = sp.nsimplify(dcoeff)
    g_coeff = sp.gcd(sp.Integer(1) if not num_coeff_reduced.is_Integer else num_coeff_reduced,
                      sp.Integer(1) if not den_coeff_reduced.is_Integer else den_coeff_reduced)
    if g_coeff != 0 and g_coeff != 1 and num_coeff_reduced.is_Integer and den_coeff_reduced.is_Integer:
        num_coeff_reduced = num_coeff_reduced / g_coeff
        den_coeff_reduced = den_coeff_reduced / g_coeff

    def build(coeff, fdict):
        # sp.Mul(evaluate=False) обязателен: обычное умножение sympy САМО
        # распределяет число*(Add) -> раскрытую сумму (2*(a-b) -> 2*a-2*b)
        # прямо на уровне арифметики, теряя красивый факторизованный вид.
        parts = []
        if coeff != 1:
            parts.append(sp.nsimplify(coeff))
        for f, p in fdict.items():
            parts.append(f**p if p != 1 else f)
        if not parts:
            return sp.Integer(1)
        if len(parts) == 1:
            return parts[0]
        return sp.Mul(*parts, evaluate=False)

    result_num = build(num_coeff_reduced, nd)
    result_den = build(den_coeff_reduced, dd)

    common_str = None
    if common:
        # Каждый множитель-Add оборачиваем в скобки, иначе "a-3" и "a+b"
        # склеенные через · дают нечитаемое "a - 3·a + b" (выглядит как
        # единая сумма, а не произведение двух скобок).
        def _fac_piece(f, p):
            base = f"({_fmt(f)})" if f.is_Add else _fmt(f)
            return base + (f"^{p}" if p != 1 else "")
        common_str = "·".join(_fac_piece(f, p) for f, p in common)

    return result_num, result_den, common_str


def _real_roots(factor_expr, var):
    if not factor_expr.free_symbols:
        return []
    roots_set = sp.solveset(sp.Eq(factor_expr, 0), var, domain=sp.S.Reals)
    if roots_set.is_FiniteSet:
        return sorted(list(roots_set), key=lambda r: sp.N(r))
    return []

def _collect_odz(*denominators):
    all_vars = set()
    for d in denominators:
        all_vars |= d.free_symbols
    if not all_vars:
        return None
    parts, seen = [], set()
    for d in denominators:
        if not d.free_symbols:
            continue
        for factor, power in sp.factor_list(d)[1]:
            if not factor.free_symbols:
                continue
            var = list(factor.free_symbols)[0]
            for r in _real_roots(factor, var):
                key = (str(var), sp.simplify(r))
                if key in seen:
                    continue
                seen.add(key)
                parts.append(f"{var}≠{_fmt(r)}")
    if not parts:
        return "знаменатель не обращается в нуль ни при каком действительном значении переменной"
    return ", ".join(parts)


class StepBuilder:
    """Нумерует шаги последовательно (только реально добавленные)."""
    def __init__(self):
        self.steps = []
        self._n = 0
    def add(self, text):
        self._n += 1
        self.steps.append(f"Шаг {self._n}. {text}")
    def raw(self, text):
        self.steps.append(text)


# ---------------------------------------------------------------------------
# 1. СОКРАТИТЬ ДРОБЬ
# ---------------------------------------------------------------------------
def solve_reduce(num_str, den_str):
    try:
        num, den = _parse(num_str), _parse(den_str)
    except Exception as e:
        return {"error": f"Не удалось разобрать выражение: {e}"}

    sb = StepBuilder()
    sb.raw(f"Задание: сократите дробь {_frac_str(num, den)}")

    num_f = sp.factor(num)
    den_f = sp.factor(den)

    if num_f != num:
        sb.add(f"Разложим числитель на множители: {_fmt(num)} = {_fmt(num_f)}")
    else:
        sb.add(f"Числитель {_fmt(num)} не раскладывается дальше.")

    if den_f != den:
        sb.add(f"Разложим знаменатель на множители: {_fmt(den)} = {_fmt(den_f)}")
    else:
        sb.add(f"Знаменатель {_fmt(den)} не раскладывается дальше.")

    result_num, result_den, common = _cancel_via_factors(num, den)
    if common:
        sb.add(f"Сокращаем на общий множитель {common}.")
    else:
        sb.add("Общих множителей нет — дробь уже несократима.")

    answer = _frac_str(result_num, result_den)
    odz = _collect_odz(den)
    return {"steps": sb.steps, "odz": odz, "answer": answer}


# ---------------------------------------------------------------------------
# 2. СЛОЖИТЬ / ВЫЧЕСТЬ ДРОБИ (2 или 3 слагаемых)
# ---------------------------------------------------------------------------
def solve_addsub(terms):
    try:
        parsed = [{"num": _parse(t["num"]), "den": _parse(t["den"]), "sign": t.get("sign", "+")} for t in terms]
    except Exception as e:
        return {"error": f"Не удалось разобрать выражение: {e}"}

    sb = StepBuilder()
    task_txt = ""
    for i, t in enumerate(parsed):
        piece = _frac_str(t["num"], t["den"])
        if i == 0:
            task_txt += ("− " if t["sign"] == "-" else "") + piece
        else:
            task_txt += (" + " if t["sign"] == "+" else " − ") + piece
    sb.raw(f"Задание: выполните действие {task_txt}")

    den_factored = [sp.factor(t["den"]) for t in parsed]
    any_factored = False
    for i, (t, df) in enumerate(zip(parsed, den_factored), start=1):
        if df != t["den"]:
            sb.add(f"Разложим знаменатель дроби {i} на множители: {_fmt(t['den'])} = {_fmt(df)}")
            any_factored = True

    lcd = den_factored[0]
    for df in den_factored[1:]:
        lcd = sp.lcm(lcd, df)
    lcd = sp.factor(lcd)
    sb.add(f"Наименьший общий знаменатель (НОЗ): {_fmt(lcd)}")

    numerator_sum = sp.Integer(0)
    mult_strs = []
    for t in parsed:
        extra = sp.cancel(lcd / t["den"])
        mult_strs.append(f"({_fmt(t['num'])})·({_fmt(extra)})")
        term_val = sp.expand(t["num"] * extra)
        numerator_sum = numerator_sum + term_val if t["sign"] == "+" else numerator_sum - term_val

    combo_txt = mult_strs[0]
    for t, m in zip(parsed[1:], mult_strs[1:]):
        combo_txt += (" + " if t["sign"] == "+" else " − ") + m
    sb.add(f"Приводим каждую дробь к знаменателю {_fmt(lcd)} и складываем числители: {combo_txt}")

    numerator_expanded = sp.expand(numerator_sum)
    sb.add(f"Раскрываем скобки в числителе: {_fmt(numerator_expanded)}")

    result_num, result_den, common = _cancel_via_factors(numerator_expanded, lcd)
    if common:
        sb.add(f"Сокращаем дробь на общий множитель {common}.")

    answer = _frac_str(result_num, result_den)
    odz = _collect_odz(*[t["den"] for t in parsed])
    return {"steps": sb.steps, "odz": odz, "answer": answer}


# ---------------------------------------------------------------------------
# 3. УМНОЖИТЬ / РАЗДЕЛИТЬ ДВЕ ДРОБИ
# ---------------------------------------------------------------------------
def solve_muldiv(num1_str, den1_str, op, num2_str, den2_str):
    try:
        num1, den1 = _parse(num1_str), _parse(den1_str)
        num2, den2 = _parse(num2_str), _parse(den2_str)
    except Exception as e:
        return {"error": f"Не удалось разобрать выражение: {e}"}

    sb = StepBuilder()
    op_sym = "·" if op == "*" else ":"
    sb.raw(f"Задание: выполните действие {_frac_str(num1,den1)} {op_sym} {_frac_str(num2,den2)}")

    # ОДЗ для деления: нужны den1≠0, den2≠0 (вторая дробь определена) И
    # num2≠0 (делить на нуль нельзя, а после переворота num2 становится
    # новым знаменателем) — все три условия считаем ДО переворота дроби.
    odz_dens = [den1, den2]
    if op == "/":
        odz_dens.append(num2)
        sb.add(f"Деление заменяем умножением на дробь, обратную второй: {_frac_str(num1,den1)} · {_frac_str(den2,num2)}")
        num2, den2 = den2, num2

    n1f, d1f, n2f, d2f = sp.factor(num1), sp.factor(den1), sp.factor(num2), sp.factor(den2)
    sb.add(f"Раскладываем на множители: числитель 1 = {_fmt(n1f)}; знаменатель 1 = {_fmt(d1f)}; "
           f"числитель 2 = {_fmt(n2f)}; знаменатель 2 = {_fmt(d2f)}")

    full_num = n1f * n2f
    full_den = d1f * d2f
    result_num, result_den, common = _cancel_via_factors(full_num, full_den)
    if common:
        sb.add(f"Сокращаем на общий множитель {common}.")
    else:
        sb.add("Общих множителей нет, дробь уже несократима.")

    answer = _frac_str(result_num, result_den)
    odz = _collect_odz(*odz_dens)
    return {"steps": sb.steps, "odz": odz, "answer": answer}


# ---------------------------------------------------------------------------
# 4. РЕШИТЬ УРАВНЕНИЕ
# ---------------------------------------------------------------------------
def solve_equation(eq_str):
    try:
        if "=" not in eq_str:
            raise ValueError("в уравнении должен быть знак =")
        left_s, right_s = eq_str.split("=", 1)
        left, right = _parse(left_s), _parse(right_s)
    except Exception as e:
        return {"error": f"Не удалось разобрать уравнение: {e}"}

    sb = StepBuilder()
    sb.raw(f"Задание: решите уравнение {_fmt(left)} = {_fmt(right)}")

    free_vars = left.free_symbols | right.free_symbols
    if len(free_vars) != 1:
        return {"error": f"Поддерживается только одна переменная (найдено: {[str(v) for v in free_vars]})"}
    var = list(free_vars)[0]

    # ОДЗ считаем по знаменателям ИСХОДНЫХ левой и правой частей, а не по
    # результату sp.together(left-right): together() может полностью
    # сократить общий знаменатель (например x/(x-2)=2/(x-2)+1 даёт diff=0/1),
    # и тогда ограничение x≠2 потеряется, хотя обе части исходно не
    # определены при x=2.
    left_den = sp.fraction(sp.together(left))[1]
    right_den = sp.fraction(sp.together(right))[1]
    odz_dens = [d for d in (left_den, right_den) if d != 1]
    has_fraction = bool(odz_dens)
    odz = _collect_odz(*odz_dens) if has_fraction else None

    diff = sp.together(left - right)
    numer, denom = sp.fraction(diff)
    numer = sp.expand(numer)

    if has_fraction:
        sb.add(f"Переносим всё в левую часть и приводим к общему знаменателю: {_frac_str(numer, sp.factor(denom))} = 0")
        if numer == 0:
            sb.raw(f"Числитель тождественно равен нулю — равенство верно при ВСЕХ допустимых значениях {var} (см. ОДЗ).")
            return {"steps": sb.steps, "odz": odz, "answer": f"верно при любом {var}, кроме указанных в ОДЗ"}
        if not numer.free_symbols:
            sb.raw(f"Числитель — ненулевая константа ({_fmt(numer)}) — равенство неверно ни при каком {var}.")
            return {"steps": sb.steps, "odz": odz, "answer": "корней нет"}
        sb.add(f"Дробь равна нулю, когда числитель равен нулю (а знаменатель — нет): {_fmt(numer)} = 0")
    else:
        sb.add(f"Переносим всё в левую часть: {_fmt(numer)} = 0")
        if numer == 0:
            return {"steps": sb.steps, "odz": None, "answer": f"верно при любом {var}"}
        if not numer.free_symbols:
            return {"steps": sb.steps, "odz": None, "answer": "корней нет"}

    poly = sp.Poly(numer, var)
    deg = poly.degree()

    if deg == 1:
        a_, b_ = poly.all_coeffs()
        root = sp.simplify(-b_/a_)
        sb.add(f"Линейное уравнение {_fmt(a_)}·{var}+({_fmt(b_)})=0, откуда {var} = {_fmt(root)}.")
        roots = [root]
    elif deg == 2:
        a_, b_, c_ = poly.all_coeffs()
        D = sp.expand(b_**2 - 4*a_*c_)
        sb.add(f"Квадратное уравнение: a={_fmt(a_)}, b={_fmt(b_)}, c={_fmt(c_)}. Дискриминант D=b²−4ac = {_fmt(D)}.")
        Dval = sp.simplify(D)
        if Dval.is_number and Dval < 0:
            sb.add("D<0, действительных корней нет.")
            roots = []
        else:
            sqrtD = sp.sqrt(Dval)
            x1 = sp.simplify((-b_ - sqrtD) / (2*a_))
            x2 = sp.simplify((-b_ + sqrtD) / (2*a_))
            sb.add(f"{var} = (−b±√D)/(2a) = ({_fmt(-b_)}±{_fmt(sqrtD)})/({_fmt(2*a_)})")
            roots = [x1] if x1 == x2 else [x1, x2]
    else:
        try:
            roots = sp.solve(sp.Eq(numer, 0), var)
            sb.add(f"Уравнение степени {deg}, решаем численно/через встроенный решатель.")
        except Exception as e:
            return {"error": f"Не удалось решить уравнение степени {deg}: {e}"}

    if has_fraction and roots:
        odz_expr = sp.Mul(*odz_dens) if len(odz_dens) > 1 else odz_dens[0]
        good = []
        for r in roots:
            if sp.simplify(odz_expr.subs(var, r)) != 0:
                good.append(r)
            else:
                sb.raw(f"Корень {var}={_fmt(r)} посторонний (обращает знаменатель в нуль) — отбрасываем.")
        roots = good

    try:
        roots = sorted(roots, key=lambda r: sp.N(r))
    except Exception:
        pass

    answer = "корней нет" if not roots else ", ".join(f"{var}={_fmt(r)}" for r in roots)
    return {"steps": sb.steps, "odz": odz, "answer": answer}


# ---------------------------------------------------------------------------
# JSON-обёртка для вызова из JS
# ---------------------------------------------------------------------------
def solve_json(kind, payload_json):
    payload = json.loads(payload_json)
    try:
        if kind == "reduce":
            res = solve_reduce(payload["num"], payload["den"])
        elif kind == "addsub":
            res = solve_addsub(payload["terms"])
        elif kind == "muldiv":
            res = solve_muldiv(payload["num1"], payload["den1"], payload["op"], payload["num2"], payload["den2"])
        elif kind == "equation":
            res = solve_equation(payload["eq"])
        else:
            res = {"error": f"Неизвестный тип задания: {kind}"}
    except Exception as e:
        res = {"error": f"Внутренняя ошибка: {type(e).__name__}: {e}"}
    return json.dumps(res, ensure_ascii=False)
