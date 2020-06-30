import unittest
import torch
import utils
class TestUtils(unittest.TestCase):
    def testCat(self):
        a = torch.ones((3, 1, 1 ,100,10))
        b = torch.zeros((100,725))
        c = utils.cat((a, b), -1)
        self.assertEqual(c.shape, (3, 1, 1, 100, 735))
    
    def testCast(self):
        a = torch.ones((3, 1, 1 ,100,10))
        b = torch.zeros((100,725))
        c = utils.cat((a, b), -1)
        self.assertEqual(c[0][0][0][0][0] , 1)

    def testCast2(self):
        b = torch.ones((3, 1, 1 ,100,10))
        a = torch.zeros((100,725))
        c = utils.cat((a, b), -1)
        self.assertEqual(c[0][0][0][0][0] , 0)

if __name__ == "__main__":
    unittest.main()
