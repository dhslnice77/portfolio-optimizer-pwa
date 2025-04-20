from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, filename):
    # 새 이미지 생성
    img = Image.new('RGB', (size, size), color='#2E86C1')
    draw = ImageDraw.Draw(img)
    
    # 텍스트 추가
    text = "PO"
    font_size = size // 3
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # 텍스트 중앙 정렬
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = (size - text_width) // 2
    y = (size - text_height) // 2
    
    # 흰색 텍스트 그리기
    draw.text((x, y), text, font=font, fill='white')
    
    # 이미지 저장
    img.save(f'static/{filename}')

# 두 가지 크기의 아이콘 생성
create_icon(192, 'icon-192.png')
create_icon(512, 'icon-512.png')
