# 导入pymysql模块，并将其安装为MySQLdb，以便Flask可以使用MySQL数据库
from email.quoprimime import unquote

import pymysql
pymysql.install_as_MySQLdb()

# 导入Flask框架及其相关模块
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

import os
from werkzeug.utils import secure_filename
import subprocess

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

from datetime import datetime
import base64

import time
import sys
from os.path import abspath, dirname
# 获取当前文件（app.py）的绝对路径
current_dir = dirname(abspath(__file__))
# 将 static 目录添加到系统路径
static_dir = os.path.join(current_dir, 'static')
sys.path.insert(0, static_dir)  # 使用 insert(0) 确保优先搜索
# 现在可以正确导入
from benchmark import auto_select, current_progress



# 初始化Flask应用
app = Flask(__name__, static_folder='static')
# 设置Flask应用的密钥，用于保护会话和其他安全功能
app.config['SECRET_KEY'] = 'your_secret_key'
# 配置数据库连接，使用MySQL数据库，连接信息包括用户名、密码、主机和数据库名
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:zhang123@127.0.0.1:3306/flaskappdb'

app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
# 修改密钥为复杂字符串（关键配置）
app.config['SECRET_KEY'] = 'your_secure_key_here_32bytes_long'

# 初始化SQLAlchemy对象，用于操作数据库
db = SQLAlchemy(app)

# 分类模型
model = ResNet50(weights='imagenet')

from flask_session import Session
# 在应用初始化后添加会话配置
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(app.root_path, 'session_cache')
Session(app)  # 初始化会话扩展

# 定义一个用户模型类，继承自db.Model
class User(db.Model):
    # 定义用户表的字段
    id = db.Column(db.Integer, primary_key=True)  # 主键
    username = db.Column(db.String(20), unique=True, nullable=False)  # 用户名，唯一且不能为空
    password = db.Column(db.String(255), nullable=False)  # 密码，不能为空

class ImageLibrary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(50), nullable=False)

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    library_id = db.Column(db.Integer, db.ForeignKey('image_library.id'), nullable=False)
    filename = db.Column(db.String(100), nullable=False)
    label = db.Column(db.String(100), nullable=True)  # 标签字段
    is_deleted = db.Column(db.Boolean, default=False)  # 标记是否被删除
    position=db.Column(db.Integer,default=0) # 排序
    upload_time = db.Column(db.DateTime, default=datetime.utcnow) # 创建时间

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# 定义首页路由，返回首页模板
@app.route("/")
def index():
    return render_template('index.html')

# 定义注册页面路由，支持GET和POST请求
@app.route("/register", methods=['GET', 'POST'])
def register():
    # 如果是POST请求，处理表单提交
    if request.method == 'POST':
        username = request.form['username']  # 获取表单中的用户名
        password = request.form['password']  # 获取表单中的密码
        confirm_password = request.form['confirm_password']  # 获取表单中的确认密码

        # 检查两次输入的密码是否一致
        if password != confirm_password:
            flash('Passwords do not match', 'danger')  # 提示密码不匹配
            return redirect(url_for('register'))  # 重定向到注册页面

        # 检查用户名是否已存在
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists', 'danger')  # 提示用户名已存在
            return redirect(url_for('register'))  # 重定向到注册页面

        # 对密码进行哈希加密
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        # 创建新用户对象
        new_user = User(username=username, password=hashed_password)
        # 将新用户添加到数据库会话
        db.session.add(new_user)
        # 提交会话，保存新用户到数据库
        db.session.commit()

        # flash('Account created successfully!', 'success')  # 提示注册成功

        user = User.query.filter_by(username=username).first()
        # 自动创建图像库
        libraries = ImageLibrary.query.filter_by(user_id=user.id).all()
        if not libraries:
            default_library = ImageLibrary(user_id=user.id, name="Default Library")
            db.session.add(default_library)
            db.session.commit()

        return redirect(url_for('register_success'))  # 重定向到注册成功页面
    # 如果是GET请求，直接返回注册页面模板
    return render_template('register.html')

# 定义注册成功页面路由
@app.route("/register_success")
def register_success():
    return render_template('register_success.html')

# 定义登录页面路由，支持GET和POST请求
@app.route("/login", methods=['GET', 'POST'])
def login():
    # 如果是POST请求，处理表单提交
    if request.method == 'POST':
        username = request.form['username']  # 获取表单中的用户名
        password = request.form['password']  # 获取表单中的密码
        user = User.query.filter_by(username=username).first()  # 查询用户是否存在

        # 如果用户不存在
        if not user:
            flash('用户不存在', 'danger')  # 提示用户名不存在
            return redirect(url_for('login'))  # 重定向到登录页面

        # 如果密码不正确
        if not check_password_hash(user.password, password):
            flash('密码错误', 'danger')  # 提示密码错误
            return redirect(url_for('login'))  # 重定向到登录页面

        # 如果验证通过，将用户ID存入会话
        session['user_id'] = user.id
        # flash('You have been logged in!', 'success')  # 提示登录成功
        return redirect(url_for('login_success'))  # 重定向到登录成功页面
    # 如果是GET请求，直接返回登录页面模板
    return render_template('login.html')

# 定义登录成功页面路由
@app.route("/login_success")
def login_success():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    libraries = ImageLibrary.query.filter_by(user_id=user_id).all()
    if not libraries:
        return redirect(url_for('login'))

    library_id = libraries[0].id  # 获取第一个库的 ID
    return render_template('login_success.html', library_id=library_id)

# 定义登出页面路由
@app.route("/logout")
def logout():
    session.pop('user_id', None)  # 从会话中移除用户ID
    flash('用户已登出!', 'success')  # 提示用户已登出
    return redirect(url_for('index'))  # 重定向到首页
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route("/super_resolution/<int:library_id>", methods=['GET', 'POST'])
def super_resolution(library_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # 验证库所有权
    library = ImageLibrary.query.filter_by(
        id=library_id,
        user_id=session['user_id']
    ).first_or_404()


    if request.method == 'POST':
        print("----图像处理请求开始----")

        # 初始化变量
        filename = None
        file_path = None

        preserved_file = request.form.get('preserved_file')
        if preserved_file:
            safe_filename = secure_filename(preserved_file)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
            if os.path.exists(file_path):
                # 直接使用已有文件
                filename = safe_filename
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                # 增强路径安全检查
                if os.path.realpath(file_path).startswith(os.path.realpath(app.config['UPLOAD_FOLDER']))and os.path.exists(file_path):
                    filename = safe_filename
                else:
                    return jsonify({'success': False, 'message': '保留文件无效或已过期'})

        # 情况1：处理新上传文件 -------------------------------------------------
        if not filename and 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                if not allowed_file(file.filename):
                    return jsonify({'success': False, 'message': '文件类型不支持'})

                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

        # 情况2：使用历史文件 ---------------------------------------------------
        if not filename:  # 如果没有新文件
            # 优先从表单获取last_file参数（关键修正）
            last_file = request.form.get('last_file') or request.args.get('last_file')

            if last_file:
                # 安全验证文件名
                safe_filename = secure_filename(last_file)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)

                if not os.path.exists(file_path):
                    return jsonify({'success': False, 'message': '历史文件已失效'})
                filename = safe_filename
            else:
                return jsonify({'success': False, 'message': '请选择需要处理的文件'})

        # -------------------------------------------
        original_image = os.path.join('uploads', filename).replace(os.sep, '/')
        print("使用的文件路径:", original_image)

        # 获取用户参数（噪声等级、模型等）
        noise_level = request.form.get('noise_level', '1')
        selected_model = request.form.get('model', 'VGG7')
        selected_extension = request.form.get('extension', 'png')
        selected_method = request.form.get('method', 'scale')
        scale_ratio = request.form.get('scale_ratio', '2')

        # 模型处理逻辑
        try:
            output_filename = f"output_{filename.rsplit('.', 1)[0]}.{selected_extension}"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

            # 调用waifu2x脚本
            waifu2x_script_path = os.path.join(app.root_path, 'SR', 'waifu2x.py')
            model_path = os.path.join(app.root_path, 'SR', 'models', selected_model.lower())

            result = subprocess.run(
                ['python', waifu2x_script_path,
                 '--input', file_path,
                 '--output', output_path,
                 '--noise_level', noise_level,
                 '--model_dir', model_path,
                 '--arch', selected_model,
                 '--extension', selected_extension,
                 '--method', selected_method,
                 '--scale_ratio', scale_ratio],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # 提取输出路径
            actual_output_path = None
            for line in result.stdout.split('\n'):
                if line.startswith("Output file saved to:"):
                    actual_output_path = line.split(": ")[1].strip()
                    break

            if not actual_output_path:
                return jsonify({'success': False, 'message': '无法定位输出文件'})

            # 生成返回结果
            processed_image = os.path.relpath(actual_output_path, app.config['UPLOAD_FOLDER'])
            processed_image = os.path.join('uploads', processed_image).replace(os.sep, '/')

            return jsonify({
                'success': True,
                'original_image': url_for('static', filename=original_image),
                'processed_image': url_for('static', filename=processed_image),
                'library_id': library_id
            })

        except subprocess.CalledProcessError as e:
            print("处理失败:", e.stderr)
            return jsonify({'success': False, 'message': '模型处理失败'})
        except Exception as e:
            print("未知错误:", str(e))
            return jsonify({'success': False, 'message': '服务器内部错误'})

    return render_template('super_resolution.html', library_id=library_id)

@app.route("/image_library/<int:library_id>")
def image_library(library_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # 验证库属于当前用户
    library = ImageLibrary.query.filter_by(
        id=library_id,
        user_id=session['user_id']
    ).first_or_404()

    images = Image.query.filter_by(
        library_id=library_id,
        is_deleted=False
    ).order_by(Image.position.asc()).all()

    return render_template('image_library.html',
                           library_id=library_id,
                           images=images)


@app.route('/create_library', methods=['POST'])
def create_library():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    name = request.form.get('name')
    if not name:
        return "库名称不能为空", 400

    user_id = session['user_id']
    new_library = ImageLibrary(user_id=user_id, name=name)
    db.session.add(new_library)
    db.session.commit()

    return '', 204  # 空响应配合前端reload

@app.route('/upload_image/<int:library_id>', methods=['POST'])
def upload_image(library_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    files = request.files.getlist('file')
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # 使用ResNet50模型对图像进行分类
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img) # 将加载的图像转换为NumPy数组
            img_array = np.expand_dims(img_array, axis=0) # 在图像数组的第0维（即最前面）添加一个额外的维度，这是因为模型的输入需要是一个4维数组（batch_size, height, width, channels），而我们这里只有一个图像，所以需要添加一个维度来表示batch_size
            img_array = preprocess_input(img_array) # 对图像数组进行预处理，使其符合ResNet50模型的输入要求
            # 使用加载的ResNet50模型对预处理后的图像进行预测
            # 模型会输出一个概率分布，表示图像属于各个类别的概率
            predictions = model.predict(img_array)
            decoded_predictions = decode_predictions(predictions, top=1)[0] # 将预测结果解码为人类可读的标签
            label = decoded_predictions[0][1]  # 获取最可能的分类标签

            # 保存图像和标签到数据库
            new_image = Image(library_id=library_id, filename=filename, label=label)
            db.session.add(new_image)
    db.session.commit()
    return redirect(url_for('image_library',library_id=library_id))


@app.route('/delete_image', methods=['POST'])
def delete_image():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # 获取 library_id 用于重定向
    library_id = request.form.get('library_id')
    # 获取前端传来的图像 ID 列表
    image_ids_str = request.form.get('image_ids')  # 获取字符串形式的图像 ID 列表
    if not image_ids_str:
        flash('No images selected for deletion', 'warning')
        return redirect(url_for('image_library', library_id=library_id))

    # 遍历所有图像 ID，删除对应的图像文件和数据库记录
    image_ids = image_ids_str.split(',')
    for image_id in image_ids:
        image = Image.query.get_or_404(image_id)
        image.is_deleted = True  # 标记为删除
        db.session.commit()

    flash('Selected images have been moved to the recycle bin', 'success')
    return redirect(url_for('image_library', library_id=request.form.get('library_id')))

# 添加修改标签的路由
@app.route('/edit_label/<int:image_id>', methods=['POST'])
def edit_label(image_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    new_label = request.form.get('new_label')
    image = Image.query.get_or_404(image_id)
    image.label = new_label
    db.session.commit()
    return {'success': True}

# 添加保存到图像库的路由
@app.route('/save_to_library', methods=['POST'])
def save_to_library():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    data = request.get_json()
    image_path = data.get('image_path')
    library_id = data.get('library_id')

    if not image_path or not library_id:
        return jsonify({'success': False, 'message': 'Missing parameters'})

    # 验证库所有权
    library = ImageLibrary.query.filter_by(
        id=library_id,
        user_id=session['user_id']
    ).first()

    if not library:
        return jsonify({'success': False, 'message': 'Invalid library'})

    # 保存图像到数据库
    filename = os.path.basename(image_path)
    new_image = Image(library_id=library.id, filename=filename, label="Processed Image")
    db.session.add(new_image)
    db.session.commit()

    return jsonify({'success': True, 'message': 'Image saved to library successfully'})

# 回收站
@app.route("/recycle_bin")
def recycle_bin():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    libraries = ImageLibrary.query.filter_by(user_id=user_id).all()
    if not libraries:
        return redirect(url_for('login'))

    library_id = libraries[0].id
    images = Image.query.filter_by(library_id=library_id, is_deleted=True).all()
    return render_template('recycle_bin.html', library_id=library_id, images=images)  # 传递 library_id

# 回收站恢复
@app.route('/restore_images', methods=['POST'])
def restore_images():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'User not logged in'})

    data = request.get_json()
    image_ids = data.get('image_ids', [])
    library_id = data.get('library_id')  # 获取 library_id

    if not library_id:
        return jsonify({'success': False, 'message': 'Library ID is missing'})

    # 恢复选中的图片
    for image_id in image_ids:
        image = Image.query.get_or_404(image_id)
        image.is_deleted = False  # 恢复图像
    db.session.commit()

    # 返回重定向 URL
    return jsonify({
        'success': True,
        'redirect_url': url_for('image_library', library_id=library_id)  # 确保传递 library_id
    })


# 回收站永久删除
@app.route('/permanently_delete_images', methods=['POST'])
def permanently_delete_images():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'User not logged in'})

    data = request.get_json()
    image_ids = data.get('image_ids', [])
    for image_id in image_ids:
        image = Image.query.get_or_404(image_id)
        try:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], image.filename))
        except FileNotFoundError:
            pass
        db.session.delete(image)
    db.session.commit()
    return jsonify({'success': True})


@app.route('/manual/<int:library_id>')
def manual(library_id):
    return render_template('manual.html',library_id=library_id)


@app.route('/auto_select_best_model', methods=['POST'])
def auto_select_best_model():
    data = request.get_json()
    image_path = data.get('image_path')
    library_id = data.get('library_id')
    current_progress.update({
        'total_steps': 0,
        'completed_steps': 0,
        'current_model': None,
        'current_noise_level': None
    })

    if not image_path:
        return jsonify({'success': False, 'message': 'No image path provided'})

    base_dir = os.path.join(app.root_path, 'static', 'uploads')
    full_image_path = os.path.join(base_dir, image_path)

    if not os.path.exists(full_image_path):
        return jsonify({'success': False, 'message': f'File not found: {full_image_path}'})

    # 调用一次auto_select并存储结果到session
    best_model, best_noise_level, best_psnr, chart_path = auto_select(full_image_path)
    chart_path = chart_path.replace('\\', '/')

    # 修改session存储方式（添加持久化标记）
    session.permanent = True  # 设置会话持久化
    session['auto_select_results'] = {
        'best_model': best_model,
        'best_noise_level': best_noise_level,
        'best_psnr': best_psnr,
        'chart_path': chart_path,
        'image_path': image_path,
        'library_id': library_id
    }
    # 显式保存session（关键修正）
    session.modified = True

    chart_url = url_for('static', filename=chart_path)
    return jsonify({
        'success': True,
        'chart_path': chart_url,
        'redirect_url': url_for('auto_select_route')
    })

@app.route("/auto_select")
def auto_select_route():
    # 修改为获取而不删除session数据（关键修正）
    results = session.get('auto_select_results')
    if not results:
        return "请求数据已过期或不存在", 400

    return render_template('auto_select.html',
                           best_model=results['best_model'],
                           best_noise_level=results['best_noise_level'],
                           best_psnr=results['best_psnr'],
                           chart_url=results['chart_path'],
                           library_id=results['library_id'],
                           image_path=results['image_path'])


# 显示我的库页面
@app.route("/mylibrary")
def my_library():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    libraries = ImageLibrary.query.filter_by(user_id=user_id).all()
    return render_template('mylibrary.html', libraries=libraries)

# 删除图像库路由
@app.route('/delete_library/<int:library_id>', methods=['POST'])
def delete_library(library_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '未登录'})

    library = ImageLibrary.query.get_or_404(library_id)
    if library.user_id != session['user_id']:
        return jsonify({'success': False, 'message': '无权限操作'})

    # 删除关联图片（可选）
    Image.query.filter_by(library_id=library_id).delete()

    db.session.delete(library)
    db.session.commit()
    return jsonify({'success': True})


@app.route("/image_detail/<int:image_id>")
def image_detail(image_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # 验证图片是否属于当前用户
    image = db.session.query(Image).join(ImageLibrary).filter(
        Image.id == image_id,
        ImageLibrary.user_id == session['user_id']
    ).first_or_404()
    library_id = image.library_id  # 获取图片所属的库ID
    # 获取图片元数据
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)

    # 使用Pillow获取分辨率
    from PIL import Image as PILImage
    with PILImage.open(image_path) as img:
        width, height = img.size

    # 获取上传时间（需要先添加upload_time字段到Image模型）
    return render_template('image_detail.html',
                           image=image,
                           library_id=library_id,
                           width=width,
                           height=height,
                           upload_time=image.upload_time)

# 保存编辑
@app.route('/save_edited_image/<int:image_id>', methods=['POST'])
def save_edited_image(image_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '未登录'})

    # 验证图片所有权
    image = db.session.query(Image).join(ImageLibrary).filter(
        Image.id == image_id,
        ImageLibrary.user_id == session['user_id']
    ).first_or_404()

    # 获取Base64数据
    data = request.get_json()
    image_data = data.get('image_data')
    if not image_data:
        return jsonify({'success': False, 'message': '无图像数据'})

    # 生成新文件名
    file_ext = image.filename.split('.')[-1]
    new_filename = f"edited_{datetime.now().strftime('%Y%m%d%H%M%S')}_{image.filename}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)

    # 保存图像文件
    try:
        header, encoded = image_data.split(",", 1)
        binary_data = base64.b64decode(encoded)
        with open(save_path, "wb") as f:
            f.write(binary_data)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

    # 更新数据库记录
    image.filename = new_filename
    image.upload_time = datetime.utcnow()
    db.session.commit()

    return jsonify({'success': True, 'message': '保存成功'})

# 比较页面
@app.route("/comparison")
def comparison():
    original_image = request.args.get('original')
    processed_image = request.args.get('processed')
    library_id = request.args.get('library_id', type=int)  # 新增获取library_id
    return render_template('comparison.html',
                           original_image=original_image,
                           processed_image=processed_image,
                           library_id=library_id)

# 保存到指定图像库
@app.route('/get_user_libraries')
def get_user_libraries():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '未登录'})

    user_id = session['user_id']
    libraries = ImageLibrary.query.filter_by(user_id=user_id).all()
    return jsonify({
        'success': True,
        'libraries': [{'id': lib.id, 'name': lib.name} for lib in libraries]
    })


# 进度条
@app.route('/get_progress')
def get_progress():
    try:
        # 添加零值保护
        if current_progress['total_steps'] == 0:
            return jsonify({'progress': 0, 'model': None, 'noise_level': None})

        progress_percent = int((current_progress['completed_steps'] / current_progress['total_steps']) * 100)
        return jsonify({
            'progress': progress_percent,
            'model': current_progress['current_model'],
            'noise_level': current_progress['current_noise_level']
        })
    except KeyError:
        return jsonify({'progress': 0, 'model': None, 'noise_level': None})


@app.route('/introduction')
def introduction():
    return render_template('introduction.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# 主程序入口
if __name__ == '__main__':
    print("----begin----")
    # 在应用上下文中执行以下操作
    with app.app_context():
        # 清除所有表
        # db.drop_all()
        # 创建所有表
        db.create_all()
    # 启动Flask应用，开启调试模式
    app.run(debug=True)