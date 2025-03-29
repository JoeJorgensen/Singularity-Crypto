# Project Rules for CryptoTrading

## Code Style & Organization

1. **Module Structure**: Follow the established directory structure. All new modules should be placed in their appropriate directories.

2. **Naming Conventions**:
   - Class names: Use `CamelCase` (e.g., `TradingStrategy`, `RiskManager`)
   - Function names: Use `snake_case` (e.g., `calculate_position_size`, `get_technical_signals`)
   - Constants: Use `UPPER_SNAKE_CASE` (e.g., `MAX_POSITION_SIZE`, `DEFAULT_STOP_LOSS`)
   - Private methods/attributes: Prefix with underscore (e.g., `_calculate_signals`, `_validate_config`)

3. **Documentation**:
   - All public functions must have docstrings
   - Complex logic should be commented
   - Class responsibilities should be clearly documented

4. **Type Hints**: Use Python type hints for all function signatures

## Trading Engine Rules

1. **Risk Management**:
   - Never exceed maximum position size (default: 90% of portfolio)
   - Enforce maximum drawdown protection (default: 15%)
   - Always implement risk-per-trade limits (default: 2%)
   - Respect maximum trades per day (default: 5)
   - Maintain minimum risk/reward ratio (default: 1.5)

2. **Trade Execution**:
   - **CRITICAL**: Always use Alpaca's IEX and Websocket feed for market data, **never** use the SIP feed or CryptoFeed.US
   - For crypto data, always use the endpoints that automatically use the IEX feed
   - For websockets, always connect to "wss://stream.data.alpaca.markets/v1beta3/crypto/us"
   - Log all trade executions in the terminal with relevant details
   - Implement order confirmation checks
   - Handle API errors gracefully with proper retries
   - **IMPORTANT**: For crypto orders in Alpaca, only use 'market', 'limit', and 'stop_limit' order types
   - When implementing stop losses for crypto, always use 'stop_limit' with both stop_price and limit_price parameters

3. **Signal Generation**:
   - Use combined signals from technical, sentiment, and on-chain components
   - Apply signal thresholds consistently
   - Implement signal quality checks
   - Log signal generation in the terminal with component contributions

## Streamlit Application Rules

1. **UI Components**:
   - Maintain consistent color scheme and styling
   - Ensure responsive layout across desktop and tablet devices
   - Use appropriate visualizations for different data types
   - Implement intuitive user controls

2. **Performance**:
   - Cache expensive computations
   - Use efficient data structures for real-time updates
   - Implement pagination for history displays
   - Optimize chart rendering for performance

3. **Service Management**:
   - **CRITICAL**: Never run multiple Streamlit instances simultaneously
   - Always shut down previous instances before starting new ones
   - Verify API connections before enabling trading
   - Implement graceful shutdown procedures

4. **Streamlit API Usage**:
   - Always use `st.rerun()` instead of the deprecated `st.experimental_rerun()`
   - Use Streamlit caching decorators (`@st.cache_data`, `@st.cache_resource`) appropriately
   - Follow Streamlit's recommended patterns for session state management
   - Ensure all UI elements have appropriate help text

## API & External Services

1. **API Keys**:
   - Store all API keys in .env file (never commit to repository)
   - Implement key rotation procedures
   - Validate API key validity before starting trading
   - Handle API rate limits appropriately

2. **Data Sources**:
   - Implement fallback data sources where possible
   - Validate data quality before using for signals
   - Log data source issues and implement alerts in the terminal 
   - Cache frequently accessed data

## Testing & Quality Assurance

1. **Unit Tests**:
   - Write tests for all critical components
   - Implement test coverage for risk management rules
   - Test edge cases for signal generation
   - Verify order execution logic

2. **Integration Tests**:
   - Test API integrations with mock responses
   - Verify end-to-end trading workflows
   - Test configuration loading and validation
   - Validate performance metrics calculations

3. **Monitoring**:
   - Log all critical events and errors in the terminal
   - Implement performance monitoring
   - Track resource usage
   - Set up alerts for unusual behavior

## Security & Compliance

1. **Authentication**:
   - Implement secure authentication for user access
   - Apply principle of least privilege for API keys
   - Protect sensitive configuration data
   - Implement session timeouts

2. **Data Handling**:
   - Do not store sensitive user data unnecessarily
   - Encrypt sensitive information
   - Implement secure data transmission
   - Adhere to relevant financial regulations

## Development Workflow

1. **Version Control**:
   - Create feature branches for new development
   - Write descriptive commit messages
   - Review code before merging
   - Tag releases with semantic versioning

2. **Bug Fixes**:
   - Document all bugs with steps to reproduce
   - Fix related code to prevent similar issues
   - Add tests to prevent regression
   - Update documentation when fixing bugs

3. **Feature Additions**:
   - Document new features thoroughly
   - Maintain backward compatibility where possible
   - Update configuration templates
   - Add appropriate tests for new features

4. **Code Quality**:
   - Run linting before commits
   - Maintain consistent code style
   - Refactor duplicated code
   - Keep functions focused and concise

## Project-Specific Guidelines

1. **Signal Processing**:
   - Review and calibrate signal thresholds regularly
   - Document signal calculation methodology
   - Implement signal noise filtering
   - Validate signals against historical performance

2. **Risk Controls**:
   - Always verify risk calculations before trade execution
   - Implement circuit breakers for unusual market conditions
   - Apply gradual position building for large allocations
   - Maintain kill switch functionality

3. **Performance Reporting**:
   - Track key performance indicators consistently
   - Compare against benchmarks
   - Report drawdowns accurately
   - Calculate risk-adjusted returns

4. **Configuration Management**:
   - Validate all user configurations before applying
   - Provide sensible defaults
   - Document configuration options
   - Implement configuration version control

## Emergency Procedures

1. **Market Disruptions**:
   - Implement automatic trading pause during high volatility
   - Define procedures for manual intervention
   - Document recovery steps
   - Test emergency shutdown regularly

2. **Technical Failures**:
   - Define fallback procedures for API outages
   - Implement data integrity verification
   - Document disaster recovery steps
   - Test resilience regularly

These rules provide comprehensive guidance for development, maintenance, and operation of the CryptoTrading project. Adherence to these guidelines will ensure consistency, reliability, and safety in the automated trading system.

