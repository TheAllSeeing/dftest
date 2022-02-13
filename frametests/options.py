class DecoratorConfig:
    options = {}


def options(**kwargs):
    def decorator(func):
        DecoratorConfig.options[func] = kwargs
        return func

    return decorator


def declare_options(func, **kwargs):
    DecoratorConfig.options[func] = kwargs
