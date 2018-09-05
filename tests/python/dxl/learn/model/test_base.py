from dxl.learn.model.base import *
from dxl.learn.config import config_with_name, clear_config
from doufo import identity


def test_residual_config_direct():
    clear_config()
    r = Residual('res', identity, 0.1)
    assert r.config[Residual.KEYS.CONFIG.RATIO] == 0.1


def test_residual_config_proxy():
    clear_config()
    c = config_with_name('res')
    c[Residual.KEYS.CONFIG.RATIO] = 0.1
    r = Residual('res', identity)
    assert r.config[Residual.KEYS.CONFIG.RATIO] == 0.1


def test_residual_config_default():
    clear_config()
    r = Residual('res', identity)
    assert r.config[Residual.KEYS.CONFIG.RATIO] == 0.3


def test_residual_config_proxy_direct_conflict():
    clear_config()
    c = config_with_name('res')
    c[Residual.KEYS.CONFIG.RATIO] = 0.1
    r = Residual('res', identity, 0.2)
    assert r.config[Residual.KEYS.CONFIG.RATIO] == 0.2
