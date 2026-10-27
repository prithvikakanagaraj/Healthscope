# HealthScope Android APK wrapper

This Android Studio project wraps the deployed HealthScope Streamlit website in a small Android WebView app. The analysis remains on the web server; the app needs an internet connection.

## Before building
1. Deploy the `healthscope` folder to Streamlit Community Cloud using the main `README.md`.
2. Copy the deployed HTTPS URL, for example `https://your-app-name.streamlit.app`.
3. Open `HealthScopeAndroid/app/src/main/res/values/strings.xml`.
4. Replace `https://YOUR-HEALTHSCOPE-APP.streamlit.app` in `app_url` with your real deployed URL.
5. Install Android Studio with Android SDK Platform 35 and a compatible JDK (Android Studio's bundled JDK is suitable).
6. Open the `HealthScopeAndroid` folder in Android Studio and allow Gradle sync. If Android Studio asks to create/download the Gradle wrapper, accept it.
7. Choose **Build > Build Bundle(s) / APK(s) > Build APK(s)**.
8. When the build succeeds, use **Locate** in the build notification. The debug APK is usually under `app/build/outputs/apk/debug/app-debug.apk`.
9. Transfer that APK to your Android phone and install it. Android may ask you to allow installation from that source.

## Important
- This is source code for the wrapper, not a precompiled APK.
- Do not build while the placeholder URL is still present.
- The website must be deployed and reachable first.
- Test login/session, dataset upload, charts and downloads on the target phone. WebView behaviour can vary by Android version.
- The app does not scrape Amazon or supply real product data by itself. It uses the dataset selected in HealthScope.
