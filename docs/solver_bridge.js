// Мост между UI и Python-решателем (solver.py), запускаемым в Pyodide.
// Pyodide инициализируется ЛЕНИВО (только при первом обращении к новой
// функции), чтобы не замедлять открытие основного решебника.

let _pyodideInstance = null;
let _pyodideLoading = null;
let _solverLoaded = false;

async function getPyodideReady(onStatus) {
  if (_pyodideInstance && _solverLoaded) return _pyodideInstance;
  if (_pyodideLoading) return _pyodideLoading;

  _pyodideLoading = (async () => {
    if (typeof loadPyodide !== 'function') {
      throw new Error('LOAD_FAILED');
    }
    onStatus && onStatus('Загружаем вычислительный движок (первый раз ~2–3 сек)…');
    let pyodide;
    try {
      pyodide = await loadPyodide({ indexURL: './pyodide/' });
    } catch (e) {
      throw new Error('LOAD_FAILED');
    }
    onStatus && onStatus('Загружаем библиотеку символьной алгебры (sympy)…');
    await pyodide.loadPackage('sympy');

    onStatus && onStatus('Загружаем модуль решателя…');
    let solverSrc;
    try {
      const resp = await fetch('./solver.py');
      if (!resp.ok) throw new Error('fetch failed');
      solverSrc = await resp.text();
    } catch (e) {
      throw new Error('LOAD_FAILED');
    }
    pyodide.runPython(solverSrc);

    _pyodideInstance = pyodide;
    _solverLoaded = true;
    return pyodide;
  })();

  try {
    return await _pyodideLoading;
  } finally {
    _pyodideLoading = null;
  }
}

/**
 * Запускает решение. kind: 'reduce'|'addsub'|'muldiv'|'equation'
 * payload: объект с параметрами (см. solver.py::solve_json)
 * onStatus: callback(text) для промежуточных статусов загрузки
 * Возвращает {steps:[...], odz:str|null, answer:str} или {error:str}
 */
async function runSolver(kind, payload, onStatus) {
  let pyodide;
  try {
    pyodide = await getPyodideReady(onStatus);
  } catch (e) {
    return { error: 'FILE_PROTOCOL_BLOCKED' };
  }
  try {
    const solveJson = pyodide.globals.get('solve_json');
    const resultStr = solveJson(kind, JSON.stringify(payload));
    const result = JSON.parse(resultStr);
    return result;
  } catch (e) {
    return { error: 'Ошибка вычисления: ' + (e && e.message ? e.message : String(e)) };
  }
}
