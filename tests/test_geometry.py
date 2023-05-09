import unittest
from cardvision.utils import geometry as g
from numpy.testing import assert_almost_equal

class GeometryTest(unittest.TestCase):

    def test_line_eq(self):
        A, B, C = g._line_eq(1, 2, 3, 4)
        self.assertAlmostEqual(A*1+B*2+C, 0)
        self.assertAlmostEqual(A*3+B*4+C, 0)

        A, B, C = g._line_eq(0, 0, 1, 0)
        self.assertAlmostEqual(C, 0)
        self.assertAlmostEqual(A, 0)

        A, B, C = g._line_eq(0, 0, 0, 1)
        self.assertAlmostEqual(C, 0)
        self.assertAlmostEqual(B, 0)

        A, B, C = g._line_eq(1, 1, 0, 0)
        self.assertGreater(A, 0)
        self.assertLess(B, 0)

    def test_point_side(self):
        lineDiag = g._line_eq(1, 1, 0, 0)
        self.assertGreater(g._point_side(1, 0, *lineDiag), 0)
        self.assertLess(g._point_side(0, 1, *lineDiag), 0)
        self.assertAlmostEqual(g._point_side(2, 2, *lineDiag), 0)

    def test_line_intersect(self):
        pnts = [(0, 0), (0, 1), (1, 1), (1, 0)]

        self.assertIsNone(
            g._line_intersect(
                *g._line_eq(*pnts[0], *pnts[1]),
                *g._line_eq(*pnts[2], *pnts[3]),
            )
        )

        assert_almost_equal(
            g._line_intersect(
                *g._line_eq(*pnts[0], *pnts[1]),
                *g._line_eq(*pnts[1], *pnts[2]),
            ),
            pnts[1]
        )

        assert_almost_equal(
            g._line_intersect(
                *g._line_eq(*pnts[0], *pnts[2]),
                *g._line_eq(*pnts[1], *pnts[3]),
            ),
            (0.5, 0.5)
        )

    def test_convex_clip(self):
        rectA = [(0, 0), (0, 1), (1, 1), (1, 0)]
        rectB = [(0, 0), (0, 1), (1, 0)]
        rectC = [(0, 1), (1, 1), (1, 0)]
        triA = [(0, 0), (0, 1.5), (1.5, 0)]

        out = g._convex_clip(rectA, rectB)
        assert_almost_equal(out, rectB)
        out = g._convex_clip(rectA, rectC)
        assert_almost_equal(out, rectC)
        out = g._convex_clip(rectA, triA)
        assert_almost_equal(out, [(0, 0), (0, 1), (0.5, 1), (1, 0.5), (1, 0)])

    def test_polygon_area(self):
        self.assertAlmostEqual(
        g._polygon_area([(0, 0), (0, 1), (1, 1), (1, 0)])
        ,1)
        self.assertAlmostEqual(
            g._polygon_area([(0, 0), (0, 1), (1, 0)])
        ,0.5)

    def test_iou(self):
        self.assertAlmostEqual(
            g.iou(
                [(0, 0), (0, 2), (1, 2), (1, 0)],
                [(0, 0), (0, 1), (2, 1), (2, 0)],
            ),
            1/3
        )
