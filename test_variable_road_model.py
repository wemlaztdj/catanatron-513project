import unittest
import numpy as np
from variable_road_model import minimum_roads, affordable, readiness_curve
class Tests(unittest.TestCase):
    def test_paths(self):
        # Chain 0-1-2-3-4-5. Own settlement 0, existing road 0-1.
        adj={i:set() for i in range(6)}
        for i in range(5):
            adj[i].add(i+1);adj[i+1].add(i)
        self.assertEqual(minimum_roads({0},{(0,1)},{0},{(0,1)},adj),1)
        # A side-branch settlement at 6 makes node 2 illegal without blocking travel.
        adj[6]={2};adj[2].add(6)
        self.assertEqual(minimum_roads({0},{(0,1)},{0,6},{(0,1)},adj),2)
        self.assertIsNone(minimum_roads({0},{(0,1)},{0,6},{(0,1),(2,3)},adj))
    def test_probability(self):
        self.assertTrue(affordable((7,3,1,0,0),True,(3,3,1,1,0)))
        self.assertFalse(affordable((6,3,1,0,0),True,(3,3,1,1,0)))
        z=np.zeros((13,5),int)
        np.testing.assert_allclose(readiness_curve((3,3,1,1,0),z,4,True,(3,3,1,1,0)),1)
        np.testing.assert_allclose(readiness_curve((0,0,0,0,0),z,4,True,(3,3,1,1,0)),0)
        z[6,0]=1
        np.testing.assert_allclose(readiness_curve((2,3,1,1,0),z,4,True,(3,3,1,1,0)),1-(31/36)**np.arange(5))
if __name__=="__main__":
    unittest.main()
