from src.logger import Logger


def test_logger_is_singleton_instantiation():
    dummy_project_name = "dummy1"
    dummy_config = {}

    logger1 = Logger(dummy_project_name, dummy_config)
    logger2 = Logger(dummy_project_name, dummy_config)

    assert id(logger1) == id(logger2)

    logger1.clean()
    logger2.clean(remove_logs=False)
