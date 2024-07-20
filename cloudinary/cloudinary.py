from configs.config import cloudinary
import io

def write_image_to_cloudinary(image, uuid):
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    image_bytes = buf.getvalue()
    upload_result = cloudinary.uploader.upload(image_bytes, public_id=uuid)
    return upload_result['secure_url']
