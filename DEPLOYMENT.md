# Deployment Guide

This guide covers deploying the Snowflake RAG Knowledge Base application to Streamlit Cloud.

## Prerequisites

- A GitHub account
- A Streamlit Cloud account (free tier available)
- Your code pushed to a GitHub repository

## Step-by-Step Deployment

### 1. Prepare Your Repository

Ensure your repository has:
- ✅ `app.py` (main application file)
- ✅ `requirements.txt` (with all dependencies)
- ✅ `.streamlit/config.toml` (Streamlit configuration)
- ✅ `.streamlit/secrets.toml` (local secrets, excluded via `.gitignore`)
- ✅ `.gitignore` (excluding `.streamlit/secrets.toml`)
- ✅ All necessary Python modules (`snowflake_utils.py`, `pdf_processor.py`)

### 2. Push to GitHub

```bash
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

### 3. Deploy to Streamlit Cloud

1. **Sign in to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Create New App**
   - Click "New app" button
   - Select your repository
   - Choose the branch (usually `main` or `master`)
   - Set the main file path to: `app.py`

3. **Configure Secrets**
   - Click on your app to open settings
   - Go to "Settings" → "Secrets"
   - Add your Snowflake credentials in TOML format:
   
   ```toml
   [snowflake]
   account = "your-account-identifier"
   user = "your-username"
   password = "your-password"
   role = "ACCOUNTADMIN"
   warehouse = "COMPUTE_WH"
   database = "DOCUMENT_CHUNKS_DB"
   schema = "public"
   ```
   
   **Important**: 
   - Use the exact structure shown above
   - All values should be in quotes
   - Never commit secrets to your repository

4. **Deploy**
   - Click "Save" to save your secrets
   - Click "Deploy" or wait for automatic deployment
   - The app will build and deploy automatically

### 4. Verify Deployment

Once deployed:
1. Your app will be available at: `https://your-app-name.streamlit.app`
2. Test the connection by entering a Team ID
3. Try uploading a test PDF
4. Test the query functionality

## Troubleshooting

### Build Failures

If the build fails:
- Check that `requirements.txt` includes all dependencies
- Verify Python version compatibility (3.8+)
- Check build logs in Streamlit Cloud dashboard

### Connection Errors

If you see connection errors:
- Verify secrets are correctly configured (check spelling and structure)
- Ensure Snowflake account allows connections from Streamlit Cloud IPs
- Check that warehouse, database, and schema exist
- Verify user has necessary permissions

### Import Errors

If you see import errors:
- Ensure all Python files are in the repository
- Check that file names match exactly (case-sensitive)
- Verify `requirements.txt` includes all packages

### Secrets Not Working

If secrets aren't being read:
- Verify the secrets structure matches exactly (including brackets and section name)
- Check that keys are lowercase in secrets
- Ensure you saved the secrets after editing
- Try redeploying the app

## Updating Your App

To update your deployed app:
1. Make changes to your code
2. Commit and push to GitHub
3. Streamlit Cloud will automatically detect changes and redeploy
4. Monitor the deployment in the Streamlit Cloud dashboard

## Environment-Specific Configuration

The application uses Streamlit secrets for both local and cloud environments:
- **Local**: Reads from `.streamlit/secrets.toml`
- **Streamlit Cloud**: Uses secrets configured in the dashboard

No code changes needed when switching between environments!

## Security Best Practices

1. ✅ Never commit `.streamlit/secrets.toml` to Git (add to `.gitignore`)
2. ✅ Use Streamlit Cloud secrets (configured in dashboard) for production
3. ✅ Rotate credentials periodically
4. ✅ Use least-privilege roles in Snowflake
5. ✅ Monitor access logs in Snowflake

## Support

For issues specific to:
- **Streamlit Cloud**: Check [Streamlit Cloud documentation](https://docs.streamlit.io/streamlit-cloud)
- **Snowflake**: Check [Snowflake documentation](https://docs.snowflake.com)
- **This Application**: Check the main README.md file
