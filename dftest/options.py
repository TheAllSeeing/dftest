_options_dict = {}


def get_test_options(test_func: callable):
    return _options_dict.get(test_func, None)


def options(**kwargs):
    """
    Parametric Decorator to declare test options
    :param kwargs:
    :return:
    """

    def decorator(func):
        _options_dict[func] = kwargs
        return func

    return decorator


def declare_options(func, **kwargs):
    _options_dict[func] = kwargs
