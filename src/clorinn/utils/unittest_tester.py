
import unittest
import logging
from . import project
from .logs import CustomLogger

class UnittestTester:
    """
    TextTestRunner accepts verbosity as an integer with three meaningful values:
        0 - silent, prints only the final summary line (Ran N tests in Xs)
        1 - default, prints dots for success, 
            s for skip, F for failure, E for error, plus the final summary.
        2 - verbose, prints the full test description followed by 
            ok, skipped 'reason', FAIL, or ERROR for each test. 
            This is what sets showAll=True in TextTestResult

    Any integer higher than 2 is accepted without error but behaves identically to 2.
    """
    def __init__(self, test_class, verbosity = 0):
        self.test_class = test_class
        self.loader = unittest.TestLoader()
        self.verbosity = verbosity

        if not isinstance(test_class, list):
            self.test_class = [test_class]
        else:
            self.test_class = test_class


    def _test_list(self):
        test_list = [self.loader.loadTestsFromTestCase(tclass) \
                        for tclass in self.test_class]
        return test_list


    def _test_suites(self):
        suites = unittest.TestSuite(self._test_list())
        return suites


    def execute(self):
        self.runner = unittest.TextTestRunner(
            resultclass = UnittestResult, # overrides unittest.TextTestResult
            verbosity = self.verbosity)
        self.runner.run(self._test_suites())


# class UnittestResult(unittest.TextTestResult):
#     def addSuccess(self, test):
#         super(unittest.TextTestResult, self).addSuccess(test)
#         if self.showAll:
#             self.stream.writeln("ok")
#         elif self.dots:
#             # All this for removing a dot from output :)
#             #self.stream.write('.')
#             self.stream.flush()

class UnittestResult(unittest.TextTestResult):
    """
    Override classes in the unittest to print outputs with CustomLogger.

    TextTestRunner verbosity translates to TextTestResult:
        verbosity = 2 --> showAll = True
        verbosity = 1 --> showAll = False, dots = True
        verbosity = 0 --> showAll = False, dots = False
    """

    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        level = logging.getLogger(project.get_name()).getEffectiveLevel()
        self._logger = CustomLogger(__name__, level=level)

    def getDescription(self, test):
        """
        By default, getDescription appends the first line of docstring 
        to the test.id(), joined by a newline.

        I am overriding the description to suit the need of this package.

        test.id() --> full.qualified.path 
                        e.g. clorinn.tests.test_current_behavior.TestRegressionNNM.test_M_identical
        str(test) --> test_method_name (full.qualified.path)
                        e.g. test_M_identical (clorinn.tests.test_current_behavior.TestRegressionNNM.test_M_identical)
        """
        parts = test.id().split('.')
        test_id_str = '.'.join(parts[2:])
        return test_id_str

    def startTest(self, test):
        unittest.TestResult.startTest(self, test)  # skip TextTestResult's stream write
        if self.showAll:
            self._logger.info(f"{self.getDescription(test)} ... start")

    def addSuccess(self, test):
        super(unittest.TextTestResult, self).addSuccess(test)
        if self.showAll:
            self._logger.info(f"{self.getDescription(test)} ... PASS")
        elif self.dots:
            self._logger.info(f"PASS")

    def addSkip(self, test, reason):
        super(unittest.TextTestResult, self).addSkip(test, reason)
        if self.showAll:
            self._logger.info(f"{self.getDescription(test)} ... skipped '{reason}'")
        elif self.dots:
            self._logger.info(f"skipped '{reason}'")

    def addFailure(self, test, err):
        super(unittest.TextTestResult, self).addFailure(test, err)
        self._logger.error(f"{self.getDescription(test)} ... FAIL")

    def addError(self, test, err):
        super(unittest.TextTestResult, self).addError(test, err)
        self._logger.error(f"{self.getDescription(test)} ... ERROR")


class MTestProgram(unittest.TestProgram):
    """
    Inject overriden class to the main test program.
    Usage:
        ```
        import utils.unittest_tester as m_unittest
        m_unittest.main()
        ```
    """
    def runTests(self):
        self.testRunner = unittest.TextTestRunner(resultclass = UnittestResult)
        super(MainTestProgram, self).runTests()

main = MTestProgram
