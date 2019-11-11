import enum


ID_MAP_T1H2O = 'T1H2Omap'
ID_MAP_FF = 'FFmap'
ID_MAP_B1 = 'B1map'
ID_MASK_FG = 'mask_fg'
ID_MASK_T1H2O = 'mask_t1h2o'


class FileTypes(enum.Enum):
    """Represents human readable MRF T1-FF file types."""
    Data = 1
    T1H2Omap = 2  #: T1 H2O (water) map.
    FFmap = 3  #: Fat fraction map.
    B1map = 4,  # B1 transmit field efficacy map.
    ForegroundTissueMask = 5  #: The foreground tissue mask (foreground=1, background=0).
    T1H2OTissueMask = 6  #: The tissue mask to evaluate the T1H2O map (FF < 0.65 = 1, background=0).
