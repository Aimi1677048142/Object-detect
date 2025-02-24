from tool.base_tool import get_label_name, picture_up_down

file_path_txt = r'D:\pcb_data\data1\temp_labels'
file_path_picture = r'D:\pcb_data\data1\temp_images'
category = get_label_name(r'D:\pcb_data\data1\temp_labels\classes.txt')
if not category:
    category = {x: f'category{x}' for x in range(10)}
distance = 10
font_scale = 0.5
picture_suffix = ".jpg"
file_suffix = ".txt"
save_draw_picture_path = r'D:\pcb_data\data1\images\val_draw_frame2'
picture_up_down(file_path_txt, file_path_picture, category, distance, font_scale, picture_suffix, file_suffix,
                save_draw_picture_path, 100, 2, 800)
