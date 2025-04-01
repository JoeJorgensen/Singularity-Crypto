# Cloud Performance Optimization Guide

This guide provides instructions on optimizing the CryptoTrader application for deployment on Streamlit Cloud or other cloud platforms.

## Performance Considerations

When running in cloud environments, the application automatically enables cloud-optimized logging to reduce overhead and improve performance. Here are the key optimizations:

1. **Reduced Logging**: The application detects cloud environments and minimizes logging output
2. **Deduplication**: Repeated log messages are consolidated to reduce noise
3. **Minimal File I/O**: File logging is disabled in cloud mode to reduce disk I/O
4. **Prioritized Logs**: Only important events and warnings are logged in cloud mode

## Environment Variables

To further optimize performance in cloud environments, you can set the following environment variables:

```
STREAMLIT_CLOUD=true
```

Adding this variable to your deployment ensures the application runs in cloud-optimized mode.

## Deploying to Streamlit Cloud

When deploying to Streamlit Cloud, follow these steps to ensure optimal performance:

1. **Set Secrets**:
   - Configure your Alpaca API credentials in Streamlit's secrets manager
   - Ensure all required API keys are properly set

2. **Configure Cloud Mode**:
   - Add the `STREAMLIT_CLOUD=true` environment variable in Streamlit Cloud settings

3. **Memory Settings**:
   - Consider setting a higher memory limit for your application if available
   - The default memory allocation may be sufficient for normal operation

4. **Connection Pooling**:
   - Alpaca API connections are automatically managed to prevent connection limits
   - The application implements connection pooling and rate limiting

## Local Testing of Cloud Mode

To test cloud-optimized performance locally:

```bash
# Set the environment variable
export STREAMLIT_CLOUD=true

# Run the application
streamlit run trading_app.py
```

This simulates cloud mode on your local machine, allowing you to verify performance optimizations.

## Monitoring Performance

When running in cloud mode, the application will still log critical information and warnings. You can monitor performance through:

1. The application UI - all important metrics are displayed regardless of logging level
2. Streamlit Cloud logs - accessible through the Streamlit Cloud dashboard
3. Application metrics in the "System Info" section of the UI

## Troubleshooting Cloud Performance

If you experience performance issues in cloud environments:

1. Verify the `STREAMLIT_CLOUD` environment variable is set
2. Check that memory usage is within limits
3. Monitor API rate limits and connection pools
4. Increase the refresh interval to reduce update frequency if needed

For persistent issues, you can temporarily enable more verbose logging by setting:

```
DEBUG_OVERRIDE=true
```

This will override the cloud optimization settings and enable more detailed logging for troubleshooting.

## Questions or Issues

If you have questions or encounter issues with cloud deployment, please open an issue on the project's GitHub repository. 