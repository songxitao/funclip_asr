import re
import os

def merge_srt_to_ass(ass_path, srt_translated_path, output_path):
    """
    将翻译后的 SRT 合并到带卡拉OK效果的 ASS 文件中
    """
    # 1. 读取原 ASS 内容
    with open(ass_path, 'r', encoding='utf-8') as f:
        ass_lines = f.readlines()

    # 2. 读取翻译后的 SRT 内容，解析成 {index: text}
    translations = []
    with open(srt_translated_path, 'r', encoding='utf-8') as f:
        srt_content = f.read().strip()
        # 匹配每一块内容：序号\n时间\n文本
        # 我们用更稳健的正则：找到时间行，然后取其后的内容直到下一个序号
        blocks = re.split(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', srt_content)
        translations = [b.strip() for b in blocks if b.strip()]

    # 3. 处理 ASS 头部和样式
    new_ass_lines = []
    events_started = False
    trans_style_added = False
    
    for line in ass_lines:
        # 在 [V4+ Styles] 后面插入一个新的样式用于翻译
        if "[V4+ Styles]" in line:
            new_ass_lines.append(line)
            continue
        
        if line.startswith("Style: Karaoke") or line.startswith("Style: Default"):
            # 修改原有样式：将英文缩小到 48，位置压低到 30
            # 这里的正则或者字符串替换要精准
            parts = line.split(",")
            if len(parts) > 22:
                parts[2] = "48"    # Fontsize
                parts[21] = "20"   # MarginV (把英文压到最下面)
                line = ",".join(parts)
            new_ass_lines.append(line)
            
            if not trans_style_added:
                # 插入翻译样式：中文 48 号字，位置顶到 100 (显示在英文上面)
                new_ass_lines.append("Style: Translation,微软雅黑,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,10,10,95,1\n")
                trans_style_added = True
            continue
        
        if "[Events]" in line:
            events_started = True
            new_ass_lines.append(line)
            continue
        
        # 4. 核心：在每行 Dialogue 之前插入对应的翻译行
        if events_started and line.startswith("Dialogue: "):
            # 提取原有的时间戳和基本信息
            # Dialogue: 0,0:00:01.20,0:00:04.50,Style,,0,0,0,,Text
            parts = line.split(",", 9)
            if len(parts) >= 10:
                layer, start, end, style, name, marginL, marginR, marginV, effect, text = parts
                
                # 获取对应的翻译文本 (按顺序)
                if translations:
                    trans_text = translations.pop(0)
                    # 插入翻译行 (使用 Translation 样式，无 \k 标签)
                    new_ass_lines.append(f"Dialogue: 0,{start},{end},Translation,,0,0,0,,{trans_text}\n")
                
                # 依然保留原有的行 (含卡拉OK效果)
                new_ass_lines.append(line)
            continue
            
        new_ass_lines.append(line)

    # 5. 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(new_ass_lines)
    print(f"✅ 双语 ASS 合并完成: {output_path}")

if __name__ == "__main__":
    # 使用示例
    base_dir = r"e:\FunClip\trans"
    ass_in = os.path.join(base_dir, "Principles for Dealing with the Changing World Ord_cut.ass")
    srt_in = os.path.join(base_dir, "trans.srt")
    out_ass = os.path.join(base_dir, "bilingual_output.ass")
    
    merge_srt_to_ass(ass_in, srt_in, out_ass)
