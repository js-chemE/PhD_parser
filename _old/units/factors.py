import scipy.constants as const

SI_FACTORS = {
    "energy": {
        "eV": const.electron_volt,  # 1 eV to joules,
        "kcal": 1e3 * const.calorie,  # 1 kcal to joules
        "kJ": 1e3,  # 1 kJ to joules
        "kWh": 1e3 * 60 * 60,  # 1 kWh to joules
        "Wh": 60 * 60,  # 1 Wh to joules
        "cal": const.calorie,  # 1 cal to joules
    },
    "wavenumber": {
        "1/cm": 100.0,  # 1/cm to 1/m
        "1/m": 1.0,  # 1/m to 1
    },
    "frequency": {
        "Hz": 1.0,  # Hz to Hz
        "kHz": 1e3,  # kHz to Hz
        "MHz": 1e6,  # MHz to Hz
        "GHz": 1e9,  # GHz to Hz
    },
    "length": {
        "A": 1e-10,  # Angstrom to m
        "nm": 1e-9,  # nm to m
        "um": 1e-6,  # um to m
        "mm": 1e-3,  # mm to m
        "cm": 1e-2,  # cm to m
        "dm": 1e-1,  # dm to m
        "m": 1.0,    # m to m
        "km": 1e3,   # km to m,
        "mile": 1609.34,  # mile to m
    },
}