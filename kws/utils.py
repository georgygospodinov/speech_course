import hydra
import omegaconf


def omegaconf_extension(func):
    def wrapper(*args, **kwargs):
        omegaconf.OmegaConf.register_new_resolver("len", lambda x: len(x))
        omegaconf.OmegaConf.register_new_resolver("getindex", lambda lst, idx: lst[idx])
        omegaconf.OmegaConf.register_new_resolver(
            "function", lambda x: hydra.utils.get_method(x)
        )
        func(*args, **kwargs)

    return wrapper
