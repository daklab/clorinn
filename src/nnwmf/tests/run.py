
from nnwmf.utils.unittest_tester import UnittestTester
from nnwmf import tests as mtests

import inspect

def get_test_classes(class_names = ['all']):
    test_classes = []
    for name, obj in inspect.getmembers(mtests):
        if inspect.isclass(obj):
            if 'all' in class_names:
                test_classes.append(obj)
            elif name in class_names:
                test_classes.append(obj)
    return test_classes


def run_unittests(test_class_names = None):
    # Run all tests if no class name is specified
    if test_class_names is None:
        test_class_names = ['all']
    test_classes = get_test_classes(test_class_names)
    tester = UnittestTester(test_classes)
    tester.execute()
    del tester
    # =========================
    # if you want report for each class separately,
    # =========================
    #for mtest in test_classes:
    #    tester = UnittestTester(mtest)
    #    del tester
    return
