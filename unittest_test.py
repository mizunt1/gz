import unittest
import torch
class TestUtils(unittest.TestCase):
    def testCat(self):
        a = torch.ones((3, 1, 1 ,100,10))
        b = torch.zeros((100,725))
        c = utils.cat(a, b, -1)
        self.assertEqual(c.shape, (3, 1, 1, 100, 735))
    
    def testCast(self):
        a = torch.ones((3, 1, 1 ,100,10))
        b = torch.zeros((100,725))
        c = utils.cat(a, b, -1)
        self.assertEqual(c[0][0][0][0][0] , 1)

if __name__ == "__main__":
    unittest.main()
