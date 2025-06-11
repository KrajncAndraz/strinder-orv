from flask import Flask, request, jsonify
#from face_logic import save_face_setup, verify_face_image
import subprocess # for running model training script


app = Flask(__name__)
# test for runner
# Route for 2FA face setup
@app.route('/setup-face', methods=['POST'])
def setup_face():
    data = request.get_json()
    user_id = data.get('userId')
    images = data.get('images', [])
    print("GOT REQUEST")
    if not user_id:
        return jsonify({'success': False, 'message': 'Missing userId or 5 images'}), 400

    #success = save_face_setup(user_id, images)
    success = True
    if success:
        # Call the training script as a subprocess
        subprocess.Popen(['python', 'train_user_model.py', str(user_id)])
        return jsonify({'success': True, 'message': 'Face data setup complete, training started'})
    else:
        return jsonify({'success': False, 'message': 'Setup failed'}), 500


# Route for 2FA face verification
@app.route('/verify-face', methods=['POST'])
def verify_face():
    data = request.get_json()
    user_id = data.get('userId')
    image = data.get('image')

    if not user_id or not image:
        return jsonify({'success': False, 'message': 'Missing userId or image'}), 400

    print(f"[INFO] Verifying face for user '{user_id}'")

    #success = verify_face_image(user_id, image)
    success = True
    if success:
        return jsonify({'success': True, 'message': 'Face verified'})
    else:
        return jsonify({'success': False, 'message': 'Verification failed'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
