def test_import_fastyield():
    import fastyield
    assert fastyield.__version__ is not None


def test_import_core_modules():
    from fastyield import config
    from fastyield import spectrum
    from fastyield.FastCurves import FastCurves