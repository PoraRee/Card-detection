"""
Utilities for evaluation of Card Vision System.
"""

# Bounding Box Calculation


from typing import List, Optional, Tuple


def _line_eq(xa: float, ya: float, xb: float, yb: float) -> Tuple[float, float, float]:
    """
    Reforumulate line passing the points (xa, ya) and (xb, yb) as
    Ax + By + C = 0

    return (A, B, C)
    """

    A = ya - yb
    B = xb - xa
    C = xa*yb - xb*ya
    return (A, B, C)


def _point_side(xp: float, yp: float, A: float, B: float, C: float) -> float:
    """
    Check if the point is on the right, on, or left side of the line
    """

    """
    Using _line_eq

       <0
    B- =0 -->A
       >0
    """

    return A*xp + B*yp + C


def _line_intersect(Aa: float, Ba: float, Ca: float, Ab: float, Bb: float, Cb: float) -> Optional[Tuple[float, float]]:
    """ Check if two line intersects """
    det = Aa*Bb - Ab*Ba
    if det == 0:
        return None # Parallel
    x_int = (Ba*Cb - Bb*Ca)/det
    y_int = (Ab*Ca - Aa*Cb)/det
    return x_int, y_int


def _convex_clip(target: List[Tuple[float, float]], clipper: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Clipping the target polygon with the convex clipper shape.
    the result is the target which are enclosed within the clipper
    """

    for i in range(len(clipper)):
        pnt_c1 = clipper[i]
        pnt_c2 = clipper[(i+1)%len(clipper)]
        clip_line = _line_eq(*pnt_c2, *pnt_c1)

        new_target: List[Tuple[float, float]] = []
        for j in range(len(target)):
            pnt_a = target[j]
            pnt_b = target[(j+1)%len(target)]

            region_a, region_b = _point_side(*pnt_a, *clip_line), _point_side(*pnt_b, *clip_line)

            if region_a >= 0:
                new_target.append(pnt_a)
            if (region_a > 0 and region_b < 0) or (region_a < 0 and region_b > 0):
                pnt_int = _line_intersect(*clip_line, *_line_eq(*pnt_a, *pnt_b))
                assert pnt_int
                new_target.append(pnt_int)
        target = new_target

    return target


def _polygon_area(pnts: List[Tuple[float, float]]) -> float :
    """
    Calculate area of the polygon
    """

    area_acc = 0.0

    for i in range(len(pnts)):
        xa, ya = pnts[i]
        xb, yb = pnts[(i+1)%len(pnts)]
        area_acc += xa*yb - xb*ya

    return abs(area_acc)/2


def iou(output: List[Tuple[float, float]], target: List[Tuple[float, float]]):
    """
    Calculate IoU.
    note that target must be convex. if not, the iou may be lower that what it supposed to be
    Polygon must be clockwise ordered.
    """
    o_a = _polygon_area(output)
    o_t = _polygon_area(target)
    o_i = _polygon_area(_convex_clip(output, target))

    union_a = o_a + o_t - o_i

    return o_i/union_a

