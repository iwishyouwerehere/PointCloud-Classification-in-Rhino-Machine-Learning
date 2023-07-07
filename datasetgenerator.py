
import rhinoscriptsyntax as rs
import random
import json
import os

def get_point_cloud(object_id, num_points):
    # Get the BREP bounding box
    bounding_box = rs.BoundingBox(object_id)

    if not bounding_box:
        print("Unable to determine bounding box.")
        return []

    # Create the point cloud
    point_cloud = []

    while len(point_cloud) < num_points:
        # Generate a random point within the bounding box
        point = [
            random.uniform(bounding_box[0][0], bounding_box[1][0]),
            random.uniform(bounding_box[0][1], bounding_box[3][1]),
            random.uniform(bounding_box[0][2], bounding_box[4][2]),
        ]

        # Check if the point is inside the BREP
        if rs.IsPointInSurface(object_id, point):
            point_cloud.append(point)
            #rs.AddPoint(point)  # Visualize the points in the Rhino viewport

    return point_cloud

def main():
    # Get the BREP objects from the user
    object_ids = rs.GetObjects("Select BREP objects", 16, True, True)

    if not object_ids:
        print("No objects selected.")
        return

    # Get the number of points to sample
    num_points = rs.GetInteger("Enter the number of points to sample", 60, 1)

    if not num_points:
        print("Invalid number of points.")
        return

    # Initialize output data list
    output_data = []

    for object_id in object_ids:
        if not rs.IsBrep(object_id):
            print("Selected object is not a BREP.")
            continue

        # Get the object type from user text
        object_type = rs.GetUserText(object_id, "type")
        
        if not object_type:
            print("Object type not found in user text.")
            continue

        # Get the point cloud for the object
        point_cloud = get_point_cloud(object_id, num_points)

        if not point_cloud:
            print("Unable to generate point cloud for object.")
            continue

        # Create the output dictionary
        object_data = {
            "points": [[round(p[0], 2), round(p[1], 2), round(p[2], 2)] for p in point_cloud],
            "label": object_type
        }

        # Add object data to the output list
        output_data.append(object_data)
    home_dir = os.path.expanduser("~")
    output_file_path = os.path.join(home_dir, 'output.json')
    
    with open(output_file_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        print(os.path.dirname(os.path.realpath(__file__)))

        print "printed"
        print output_data

if __name__ == "__main__":
    main()
