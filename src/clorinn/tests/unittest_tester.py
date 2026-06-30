import unittest
import logging
from clorinn.utils import project
from clorinn.utils.logs import configure_module_logger

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
        self.logger_ = configure_module_logger(__name__, verbosity=verbosity)

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


class UnittestResult(unittest.TextTestResult):
    """
    Override classes in the unittest to print outputs.

    TextTestRunner verbosity translates to TextTestResult:
        verbosity = 2 --> showAll = True
        verbosity = 1 --> showAll = False, dots = True
        verbosity = 0 --> showAll = False, dots = False
    """

    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.logger_ = logging.getLogger(__name__)

    def getDescription(self, test):
        """
        By default, getDescription appends the first line of docstring 
        to the test.id(), joined by a newline.

        I am overriding the description to suit the need of this package.

        test.id() --> full.qualified.path 
                        e.g. clorinn.tests.regression.test_current_behavior.TestRegressionFWNNM.test_M_identical
        str(test) --> test_method_name (full.qualified.path)
                        e.g. test_M_identical (clorinn.tests.regression.test_current_behavior.TestRegressionFWNNM.test_M_identical)
        """
        parts = test.id().split('.')
        #test_id_str = '.'.join(parts[2:])
        test_id_str = parts[-1]
        return test_id_str

    def test_name(self, test):
        parts = test.id().split('.')
        return {"test_name": parts[-2]}
        

    def startTest(self, test):
        unittest.TestResult.startTest(self, test)  # skip TextTestResult's stream write
        if self.showAll:
            self.logger_.info(f"{self.getDescription(test)} ... start", extra=self.test_name(test))

    def addSuccess(self, test):
        super(unittest.TextTestResult, self).addSuccess(test)
        if self.showAll:
            self.logger_.info(f"{self.getDescription(test)} ... PASS", extra=self.test_name(test))
        elif self.dots:
            self.logger_.info(f"PASS", extra=self.test_name(test))

    def addSkip(self, test, reason):
        super(unittest.TextTestResult, self).addSkip(test, reason)
        if self.showAll:
            self.logger_.info(f"{self.getDescription(test)} ... skipped '{reason}'", extra=self.test_name(test))
        elif self.dots:
            self.logger_.info(f"skipped '{reason}'", extra=self.test_name(test))

    def addFailure(self, test, err):
        super(unittest.TextTestResult, self).addFailure(test, err)
        self.logger_.error(f"{self.getDescription(test)} ... FAIL", extra=self.test_name(test))

    def addError(self, test, err):
        super(unittest.TextTestResult, self).addError(test, err)
        self.logger_.error(f"{self.getDescription(test)} ... ERROR", extra=self.test_name(test))


class MainTestProgram(unittest.TestProgram):
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

main = MainTestProgram
