
import functools

class run_once(object):
    """
    run_once enforces a class function to run only once
    during the lifetime of the object.

    Usage:

    ```
    class Foo:

        @run_once
        def bar(...):
    ```

    Warning: A function decorated with run_once is not rerun
    even if some parameters of the class is updated.
    """

    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func


    def __call__(self, instance, *args, **kwargs):
        #print (f"In run_once.__call__()")
        fname = self.func.__name__
        rname = f"{fname}_has_run"
        if not hasattr(instance, rname):
            self.func(instance, *args, **kwargs)
            setattr(instance, rname, True)
        return


    def __get__(self, instance, instancetype):
        return functools.partial(self.__call__, instance)
