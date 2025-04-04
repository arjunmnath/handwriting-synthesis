import numpy as np

def bezier_curve(t, P0, P1, P2, P3):
    """Computes a point on a cubic Bézier curve at parameter t."""
    return (1-t)**3 * P0 + 3*(1-t)**2 * t * P1 + 3*(1-t) * t**2 * P2 + t**3 * P3

def generate_gcode(bezier_curves, steps=50, feed_rate=1000):
    """
    Converts Bézier curves to G-code.
    
    bezier_curves: List of tuples, each with four control points [(P0, P1, P2, P3), ...]
    steps: Number of linear segments per curve
    feed_rate: Feed rate for G1 moves
    """
    gcode = ["G21 ; Set units to mm", "G90 ; Absolute positioning", f"F{feed_rate}"]
    
    for curve in bezier_curves:
        P0, P1, P2, P3 = map(np.array, curve)
        points = [bezier_curve(t, P0, P1, P2, P3) for t in np.linspace(0, 1, steps)]
        
        # Move to the first point
        gcode.append(f"G0 X{points[0][0]:.3f} Y{points[0][1]:.3f}")
        
        # Draw the curve with linear segments
        for pt in points[1:]:
            gcode.append(f"G1 X{pt[0]:.3f} Y{pt[1]:.3f}")
    
    gcode.append("M2 ; End of program")  # End program
    return "\n".join(gcode)

# Example: List of cubic Bézier curves
curves = [
    [(0, 0), (10, 30), (20, -30), (30, 0)],  # Example curve
    [(30, 0), (40, 40), (50, -40), (60, 0)]  # Another curve
]

gcode = generate_gcode(curves)
print(gcode)
