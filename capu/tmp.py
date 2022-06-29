import os
from gec_model import GecBERTModel
model = GecBERTModel(
 vocab_path=os.path.join("./", "vocabulary"),
 model_paths="dragonSwing/vibert-capu",
 split_chunk=True
)
model("theo đó thủ tướng dự kiến tiếp bộ trưởng nông nghiệp mỹ tom wilsack bộ trưởng thương mại mỹ gina raimondo bộ trưởng tài chính janet yellen gặp gỡ thượng nghị sĩ patrick leahy và một số nghị sĩ mỹ khác")
# Always return list of outputs.
# ['Theo đó, Thủ tướng dự kiến tiếp Bộ trưởng Nông nghiệp Mỹ Tom Wilsack, Bộ trưởng Thương mại Mỹ Gina Raimondo, Bộ trưởng Tài chính Janet Yellen, gặp gỡ Thượng nghị sĩ Patrick Leahy và một số nghị sĩ Mỹ khác.']
model("những gói cước năm g mobifone sẽ mang đến cho bạn những trải nghiệm mới lạ trên cả tuyệt vời so với mạng bốn g thì tốc độ truy cập mạng 5 g mobifone được nhận định là siêu đỉnh với mức truy cập nhanh gấp 10 lần")
# ['Những gói cước 5G MobiFone sẽ mang đến cho bạn những trải nghiệm mới lạ trên cả tuyệt vời. So với mạng 4G thì tốc độ truy cập mạng 5G MobiFone được nhận định là siêu đỉnh với mức truy cập nhanh gấp 10 lần.']