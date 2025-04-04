import cv2
import numpy as np
import scipy.optimize
import bezier
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    return binary

def extract_contours(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cnt.squeeze() for cnt in contours if len(cnt) > 5]  # Remove small/noisy contours

def fit_cubic_bezier(points):
    def bezier_curve(t, P0, P1, P2, P3):
        return (1-t)**3 * P0 + 3*(1-t)**2 * t * P1 + 3*(1-t) * t**2 * P2 + t**3 * P3
    
    t_values = np.linspace(0, 1, len(points))
    P0, P3 = points[0], points[-1]
    P1, P2 = points[len(points)//3], points[2*len(points)//3]
    
    def error_function(params):
        P1x, P1y, P2x, P2y = params
        P1, P2 = np.array([P1x, P1y]), np.array([P2x, P2y])
        fitted_points = np.array([bezier_curve(t, P0, P1, P2, P3) for t in t_values])
        return np.ravel(fitted_points - points)
    
    optimized_params = scipy.optimize.least_squares(error_function, np.ravel([P1, P2])).x
    P1, P2 = np.reshape(optimized_params, (2, 2))
    return np.array([P0, P1, P2, P3])

def extract_bezier_curves(contours):
    bezier_curves = []
    for contour in contours:
        contour = np.array(contour, dtype=np.float32)
        if len(contour) >= 4:
            bezier_curve = fit_cubic_bezier(contour)
            bezier_curves.append(bezier_curve)
    return bezier_curves

def plot_bezier_curves(bezier_curves):
    for curve in bezier_curves:
        nodes = curve.T
        bez = bezier.Curve(nodes, degree=3)
        s_vals = np.linspace(0, 1, 100)
        curve_points = bez.evaluate_multi(s_vals)
        plt.plot(curve_points[0], curve_points[1], 'r-')
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == "__main__":
    image_path = "handwriting.png"  # Change to your image
    binary_image = preprocess_image(image_path)
    contours = extract_contours(binary_image)
    bezier_curves = extract_bezier_curves(contours)
    plot_bezier_curves(bezier_curves)
