from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import Blueprint, render_template, request, flash, jsonify
from flask_login import login_required, current_user
from .models import Note
from . import db
import json
from pathlib import Path
import os

p = Path('.')
datapath = p / "AI_engine/test_data/"

views = Blueprint('views', __name__)

UPLOAD_FOLDER = datapath
ALLOWED_EXTENSIONS = {'npy'}

@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    if request.method == 'POST': 
        note = request.form.get('note')#Gets the note from the HTML 

        if len(note) < 1:
            flash('Note is too short!', category='error') 
        else:
            new_note = Note(data=note, user_id=current_user.id)  #providing the schema for the note 
            db.session.add(new_note) #adding the note to the database 
            db.session.commit()
            flash('Note added!', category='success')

    return render_template("data.html", user=current_user)


def allowed_file(filename):
    print("ALLOWED  --------------------------")
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@views.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print("IF ONE  --------------------------")
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            print("IF TWO  --------------------------")
            flash('No selected file',category="error")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            print("IF THREE  --------------------------")
            filename = secure_filename(file.filename)
            #file.save(os.path.join(views.config['UPLOAD_FOLDER'], filename))
            return filename #redirect(url_for('download_file', name=filename))
    return

@views.route('/delete-note', methods=['POST'])
def delete_note():  
    note = json.loads(request.data) # this function expects a JSON from the INDEX.js file 
    noteId = note['noteId']
    note = Note.query.get(noteId)
    if note:
        if note.user_id == current_user.id:
            db.session.delete(note)
            db.session.commit()

    return jsonify({})


@views.route('/', methods=['POST'])
def types():  
    if request.method == 'POST': 
        pass

    return render_template("types.html", user=current_user)