from waitress import serve
import ln_main_app
serve(ln_main_app.app, host='0.0.0.0', port=8000)
