from core.attacks.interfaces import TextGenerator, MultiturnStrat
from core.attacks.datastore.logger import Tag, MessageLogger, MessageType
from core.attacks.datastore.yielder_registry import YielderRegistry


import functools
import inspect
import functools


class Yielder:
    def __init__(self, listener, tag=None, use_batch=False, batch_timeout_seconds=5):
        self.listener = listener
        self.tag = tag
        self._class_cache = {}
        self.use_batch = use_batch
        self.batch_timeout_seconds = batch_timeout_seconds

    def of(self, gen):
        target_names = YielderRegistry.get_target_names(gen)
        if not target_names:
            return gen

        gen_class = type(gen)

        if gen_class not in self._class_cache:
            # 1. Creăm o clasă nouă care moștenește din clasa originală
            class DynamicYielder(gen_class):
                def _log_and_call(self, target_method_name, *args, **kwargs):
                    # Identificăm metoda originală de pe clasa părinte (super)
                    orig_method = getattr(super(), target_method_name)

                    if hasattr(self, "_reset_yield"):
                        self._reset_yield()

                    # Executăm metoda (self este chiar obiectul nostru)
                    result = orig_method(*args, **kwargs)

                    # --- Logica de logging ---
                    if hasattr(self, "_get_yield") and self._get_yield():
                        self._listener.on(
                            self._cnt,
                            self.identifier,
                            self._get_yield(),
                            self._tag,
                            MessageType.EXTRA,
                        )

                    self._listener.on(
                        self._cnt, self.identifier, result, self._tag, MessageType.RESPONSE
                    )

                    if hasattr(self, "get_last_call_params"):
                        self._listener.on(
                            self._cnt,
                            self.identifier,
                            (self.get_last_call_params(), result),
                            self._tag,
                            MessageType.CONVERSATION,
                        )

                    self._cnt += 1
                    return result

                async def _log_and_call_async(self, target_method_name, *args, **kwargs):
                    # Identificăm metoda originală de pe clasa părinte (super)
                    orig_method = getattr(super(), target_method_name)

                    if hasattr(self, "_reset_yield"):
                        self._reset_yield()

                    # Executăm metoda (self este chiar obiectul nostru)
                    result = await orig_method(*args, **kwargs)

                    # --- Logica de logging ---
                    if hasattr(self, "_get_yield") and self._get_yield():
                        self._listener.on(
                            self._cnt,
                            self.identifier,
                            self._get_yield(),
                            self._tag,
                            MessageType.EXTRA,
                        )

                    self._listener.on(
                        self._cnt, self.identifier, result, self._tag, MessageType.RESPONSE
                    )

                    if hasattr(self, "get_last_call_params"):
                        self._listener.on(
                            self._cnt,
                            self.identifier,
                            (self.get_last_call_params(), result),
                            self._tag,
                            MessageType.CONVERSATION,
                        )

                    self._cnt += 1
                    return result

            # 2. Definim metoda "wrapper" care apelează logica de logare
            def make_wrapper(m_name):
                orig_func = getattr(gen_class, m_name)

                # Check if the original method is async
                if inspect.iscoroutinefunction(orig_func):

                    @functools.wraps(orig_func)
                    async def async_wrapper(self, *args, **kwargs):
                        return await self._log_and_call_async(m_name, *args, **kwargs)

                    return async_wrapper
                else:

                    @functools.wraps(orig_func)
                    def sync_wrapper(self, *args, **kwargs):
                        return self._log_and_call(m_name, *args, **kwargs)

                    return sync_wrapper
                # @functools.wraps(getattr(gen_class, m_name))
                # def wrapper(self, *args, **kwargs):
                #     return self._log_and_call(m_name, *args, **kwargs)

                # return wrapper

            for target_name in target_names:
                setattr(DynamicYielder, target_name, make_wrapper(target_name))
            self._class_cache[gen_class] = DynamicYielder

        # 3. MAGIC: "Mutăm" obiectul existent în noua clasă fără să pierdem datele
        # Sau, dacă vrei să protejezi originalul, facem o copie superficială întâi
        import copy

        new_obj = copy.copy(gen)
        new_obj.__class__ = self._class_cache[gen_class]

        # Inițializăm variabilele de stare specifice wrapper-ului
        new_obj._listener = self.listener
        new_obj._tag = self.tag
        new_obj._cnt = 0

        return new_obj
