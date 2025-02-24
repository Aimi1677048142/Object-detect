from tool.base_tool import draw_frame_to_picture, get_label_name

file_path_txt = r'D:\pcb_data\original_ img and yolo_imfo\yolo_label_0902'
file_path_picture = r'D:\pcb_data\original_ img and yolo_imfo\original img'
category = get_label_name(r'D:\pcb_data\original_ img and yolo_imfo\original img\classes.txt')
if not category:
    category = {x: f'category{x}' for x in range(10)}
distance = 10
font_scale = 0.5
picture_suffix = ".jpg"
file_suffix = ".txt"
save_draw_picture_path = r'D:\pcb_data\original_ img and yolo_imfo\tt'
draw_frame_to_picture(file_path_txt, file_path_picture, category, distance, font_scale, picture_suffix, file_suffix,
                      save_draw_picture_path)
