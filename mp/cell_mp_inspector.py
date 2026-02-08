"""
Based on https://github.com/rusty1s/pytorch_geometric/blob/76d61eaa9fc8702aa25f29dfaa5134a169d0f1f6/torch_geometric/nn/conv/utils/inspector.py

MIT License

Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>
Copyright (c) 2021 The CWN Project Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import inspect
from collections import OrderedDict
from typing import Dict, Any, Callable, Iterable, List, Union

# PyG Inspector moved around between versions.
try:
    # Newer PyG (common in later 2.x)
    from torch_geometric.inspector import Inspector
except Exception:
    # Older PyG
    from torch_geometric.nn.conv.utils.inspector import Inspector


FuncLike = Union[Callable, str]


class CellularInspector(Inspector):
    """A thin compatibility wrapper around PyG's Inspector.

    This project historically relied on internal attributes/methods (`params`,
    `keys`, etc.) that changed across PyG versions. We therefore ensure these
    exist and behave consistently.

    Notes:
    - `params` is a dict mapping function name -> OrderedDict(signature params)
    - `inspect()` stores parsed signatures in `self.params`
    - `keys()` returns the (ordered) union of argument names across functions
    """
    def __init__(self, *args, **kwargs):
        # Eat our own kwarg (must NOT be forwarded to PyG Inspector)
        self.module = kwargs.pop("module", None)
    
        # Some PyG versions require `cls` in Inspector.__init__(cls, ...)
        cls = kwargs.pop("cls", None)
        if cls is None and self.module is not None:
            cls = self.module.__class__
    
        # Use explicit super to avoid "super(): no arguments" inside closures.
        base_init = super(CellularInspector, self).__init__
    
        last_err = None
        inited = False
    
        # Try common Inspector constructor signatures across PyG versions
        for mode in ("cls_args_kwargs", "cls_only", "args_kwargs", "empty"):
            try:
                if mode == "cls_args_kwargs":
                    if cls is None:
                        raise TypeError("cls is None")
                    base_init(cls, *args, **kwargs)
                elif mode == "cls_only":
                    if cls is None:
                        raise TypeError("cls is None")
                    base_init(cls)
                elif mode == "args_kwargs":
                    base_init(*args, **kwargs)
                else:  # "empty"
                    base_init()
    
                inited = True
                last_err = None
                break
            except TypeError as e:
                last_err = e
    
        if not inited:
            raise TypeError(
                f"Failed to initialize PyG Inspector (inferred cls={cls}). "
                f"Last TypeError: {last_err}"
            )
    
        # Ensure params exists for downstream code
        if not hasattr(self, "params") or self.params is None:
            self.params = {}



    def __implements__(self, cls, func_name: str) -> bool:
        """Checks whether `cls` implements `func_name` (excluding a specific base)."""
        if cls.__name__ == 'CochainMessagePassing':
            return False
        if func_name in cls.__dict__.keys():
            return True
        return any(self.__implements__(c, func_name) for c in cls.__bases__)

    def _resolve(self, func: FuncLike) -> Callable:
        """Resolves a function reference that may be a callable or a method name."""
        if callable(func):
            return func
        if isinstance(func, str):
            # Try to resolve the method name against `self.module` if available.
            # This mirrors how PyG Inspector sometimes resolves functions by name.
            if getattr(self, "module", None) is not None and hasattr(self.module, func):
                return getattr(self.module, func)
            # As a fallback, resolve against the inspector itself (rare but safe).
            if hasattr(self, func):
                return getattr(self, func)
            raise AttributeError(
                f"Cannot resolve function name '{func}'. "
                f"CellularInspector.module is {type(getattr(self, 'module', None))}."
            )
        raise TypeError(f"Expected a callable or str, got: {type(func)}")

    def inspect(self, func: Callable, pop_first_n: int = 0) -> Dict[str, Any]:
        """Inspects `func` signature and stores it in `self.params[func.__name__]`.

        Args:
            func: The function/method to inspect.
            pop_first_n: Number of leading parameters to drop (e.g., drop 'self').

        Returns:
            Dict-like ordered mapping of parameter name -> inspect.Parameter
        """
        # Ensure params dict exists (defensive against base-class differences).
        if not hasattr(self, "params") or self.params is None:
            self.params = {}

        params = inspect.signature(func).parameters
        params = OrderedDict(params)

        for _ in range(pop_first_n):
            if len(params) == 0:
                break
            params.popitem(last=False)

        self.params[func.__name__] = params
        return params

    def keys(self, funcs, exclude=None):
        """Return the union of argument names as a set (PyG-compatible for callers
        that use `.difference(...)`).
        """
        if exclude is None:
            exclude = []
    
        # Normalize to iterable.
        if isinstance(funcs, str) or callable(funcs):
            funcs = [funcs]
    
        out = set()
    
        for f in funcs:
            func = self._resolve(f)
            name = func.__name__
    
            if not hasattr(self, "params") or self.params is None:
                self.params = {}
    
            if name not in self.params:
                # Drop 'self'
                self.inspect(func, pop_first_n=1)
    
            for k in self.params[name].keys():
                if k in exclude:
                    continue
                out.add(k)
    
        return out

    def distribute(self, func, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Selects the subset of `kwargs` required by `func`'s signature.
    
        Args:
            func: callable or function name (str), e.g., "message_up"
            kwargs: a big collected dict (coll_dict) produced by propagate()
    
        Returns:
            A dict of arguments to pass into `func`.
        """
        fn = self._resolve(func)
        name = fn.__name__
    
        if not hasattr(self, "params") or self.params is None:
            self.params = {}
    
        if name not in self.params:
            # Drop 'self' for bound methods
            self.inspect(fn, pop_first_n=1)
    
        sig_params = self.params[name]
    
        out: Dict[str, Any] = {}
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values())
    
        if has_varkw:
            # If the function accepts **kwargs, pass everything through.
            # (But still allow explicit parameters to override naturally.)
            out.update(kwargs)
    
        for pname, p in sig_params.items():
            if p.kind == inspect.Parameter.VAR_POSITIONAL:
                # We don't support *args-style injection here.
                continue
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                continue
    
            if pname in kwargs:
                out[pname] = kwargs[pname]
            else:
                # If missing but has default, we can omit it.
                if p.default is not inspect._empty:
                    continue
                # Required argument missing -> hard error, helps debugging.
                raise KeyError(
                    f"Missing required argument '{pname}' for function '{name}'. "
                    f"Available keys: {sorted(list(kwargs.keys()))[:50]} ..."
                )
    
        return out
