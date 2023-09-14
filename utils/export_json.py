def export_json(output_matrix, num_images=30, output_json_path='/opt/app/json_out/coronary-artery-segmentation.json', mapping_dict=dict()):
    with open('/opt/app/json_out/empty_annotations.json') as file:
        gt = json.load(file)

    gt_images = [{
                "id": i+1,
                "width": 512,
                "height": 512,
                "file_name": mapping_dict.get(f"{i}.png", f"{i}.png"),
                "license": 0,
                "date_captured": iso_datetime
                }
                for i in range(num_images)]

    print('writing:',  {f'{i}.png': mapping_dict.get(f"{i}.png", f"{i}.png") for i in range(num_images)})

    info = {'description': 'NU Team 2023',
           'version': 'v1',
           'year': 2023,
           'contributor': 'txlmd',
            'date_created': datetime.today().strftime("%Y-%m-%d")}

    licenses = [{'id': i+1,
                 'name': mapping_dict.get(f"{i}.png", f"{i}.png"),
                 'url': ""} for i in range(num_images)]

    empty_submit = dict()
    empty_submit["images"] = gt_images
    empty_submit["categories"] = gt["categories"]
    empty_submit["annotations"] = []
    empty_submit["info"] = info
    empty_submit["licenses"] = licenses

    gt_mask = output_matrix[0:num_images, 1:27, ...]

    count_anns = 1
    areas = []
    for img_id, img in enumerate(gt_mask, 0):
        for cls_id, cls in enumerate(img, 0):
            contours = measure.find_contours(cls)

            for contour in contours:
                for i in range(len(contour)):
                    row, col = contour[i]
                    contour[i] = (col - 1, row - 1)

                # Simplify polygon
                poly = Polygon(contour)

                if poly.is_empty:
                    continue
                if poly.geom_type == "Polygon":
                    segmentation = np.array(poly.exterior.coords).ravel().tolist()
                elif poly.geom_type == "MultiPolygon":
                    poly = poly.simplify(1.0, preserve_topology=False)
                    segmentation = np.array(poly.exterior.coords).ravel().tolist()

                    if not poly.is_valid:
                        # Attempt to fix self-intersections using buffering
                        try:
                            buffered_poly = poly.buffer(0)
                            if buffered_poly.is_valid:
                                poly = buffered_poly
                        except:
                            raise Exception("Sorry, no numbers below zero")

                # filter out small segments
                if poly.area > 250:
                    new_ann = dict()

                    new_ann["id"] = count_anns
                    new_ann["image_id"] = img_id + 1
                    new_ann["category_id"] = cls_id + 1
                    new_ann["segmentation"] = [segmentation]
                    new_ann["area"] = poly.area
                    x, y = contour.min(axis=0)
                    w, h = contour.max(axis=0) - contour.min(axis=0)
                    new_ann["bbox"] = [int(x), int(y), int(w), int(h)]
                    new_ann["iscrowd"] = 0
                    # new_ann["attributes"] = {"occluded": False}
                    count_anns += 1
                    empty_submit["annotations"].append(new_ann.copy())
                    areas.append(poly.area)

    with open(output_json_path, "w") as file:
        json.dump(empty_submit, file)

    # print(output_json_path)
    return output_json_path
