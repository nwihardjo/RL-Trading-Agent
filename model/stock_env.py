import pandas as pd
import talib
import numpy as np
import os

class StockEnv(object):
    """
    Environment which the trading agent is trained and perform trades. Controls the flow of the one-dimentional time-series states, manages information that is passed to and from the trading agent.
    """

    def __init__(self, instruments, data_name, capital_base=1e5,
                 start_date='2015-01-01',
                 end_date=None,
                 data_local_path='./data/Processed',
                 commission_fee=5e-3,
                 normalize_length=10,
                 ):
        """
        Initialise the stock environment and its appropriate variable while clean and process the market data to be ready to be used to train the agent.
        :param instruments: list of company code which stocks are used and trained
        :param data_name: name of the pickle (serialised) file containing the stocks
        :param capital_base: initial money possessed in the unit of the stock price
        :param start_date: in the format of yyyy-mm-dd
        :param end_date: in the format of yyyy-mm-dd
        :param data_local_path: local path of which the processed and serialised data is located
        :param commission_fee: fee calculated when making a tradeonsidered in the training data
        :param normalize_length:
        """
        self.instruments = instruments
        self.capital_base = capital_base
        self.commission_fee = commission_fee
        self.normalize_length = normalize_length
        self.start_date = start_date
        self.end_date = end_date
        self.data_local_path = data_local_path
        self.preprocessed_market_data, self.cleaned_market_data = self._init_market_data(data_name = data_name)
        self.pointer = normalize_length - 1
        self.done = (self.pointer == (self.preprocessed_market_data.shape[1] - 1))

        self.current_position = np.zeros(len(self.instruments))
        self.current_portfolio_value = np.concatenate((np.zeros(len(self.instruments)), [self.capital_base]))
        self.current_weight = np.concatenate((np.zeros(len(self.instruments)), [1.]))
        self.current_date = self.preprocessed_market_data.major_axis[self.pointer]

        self.portfolio_values = []
        self.positions = []
        self.weights = []
        self.trade_dates = []

    def reset(self):
        """
        Reset the state, weights, and all variables of the environment to be the initial values
        :return: return the normalised initial state, and whether it is done or not
        """
        self.pointer = self.normalize_length
        self.current_position = np.zeros(len(self.instruments))
        self.current_portfolio_value = np.concatenate((np.zeros(len(self.instruments)), [self.capital_base]))
        self.current_weight = np.concatenate((np.zeros(len(self.instruments)), [1.]))
        self.current_date = self.preprocessed_market_data.major_axis[self.pointer]
        self.done = (self.pointer == (self.preprocessed_market_data.shape[1] - 1))

        self.portfolio_values = []
        self.positions = []
        self.weights = []
        self.trade_dates = []

        return self._get_normalized_state(), self.done

    def step(self, action):
        """
        Calculate the reward, next state, values of the stock in possession, and all relevant variables after the desired weights is returned by the agent
        :param action: matrix of the desired weights of the agent
        :return: next state, reward caused by the action taken by the agent, and indication whether the state has reached its final state or not
        """
        assert action.shape[0] == len(self.instruments) + 1
        assert np.sum(action) <= 1 + 1e5
        current_price = self.cleaned_market_data[:, :, 'adj_close'].iloc[self.pointer].values
        self._rebalance(action=action, current_price=current_price)

        self.pointer += 1
        self.done = (self.pointer == (self.preprocessed_market_data.shape[1] - 1))
        next_price = self.cleaned_market_data[:, :, 'adj_close'].iloc[self.pointer].values
        reward = self._get_reward(current_price=current_price, next_price=next_price)
        state = self._get_normalized_state()
        return state, reward, self.done

    def _rebalance(self, action, current_price):
        """
        Calculate the trade amount and keep track of the portfolio values, positions, and weights over time
        :param action: matrix of the desired amount of portfolio in possession
        :param current_price: closing price at the same state where the action is taken
        """
        target_weight = action
        target_value = np.sum(self.current_portfolio_value) * target_weight
        target_position = target_value[:-1] / current_price
        trade_amount = target_position - self.current_position
        commission_cost = np.sum(self.commission_fee * np.abs(trade_amount) * current_price)

        self.current_position = target_position
        self.current_portfolio_value = target_value - commission_cost
        self.current_weight = target_weight
        self.current_date = self.preprocessed_market_data.major_axis[self.pointer]

        self.positions.append(self.current_position.copy())
        self.weights.append(self.current_weight.copy())
        self.portfolio_values.append(self.current_portfolio_value.copy())
        self.trade_dates.append(self.current_date)

    def _get_normalized_state(self):
        """
        normalised the state between two different timepoints
        :return: normalised state appended by the weights
        """
        data = self.preprocessed_market_data.iloc[:, self.pointer + 1 - self.normalize_length:self.pointer + 1, :].values
        state = ((data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-5))[:, -1, :]
        return np.concatenate((state, self.current_weight[:-1][:, None]), axis=1)

    def get_meta_state(self):
        """"
        :return: price at the current state
        """
        return self.preprocessed_market_data.iloc[:, self.pointer, :]

    def _get_reward(self, current_price, next_price):
        """
        calculate the reward based on the amount of the stocks in possession, with respect to the price at two different states
        :param current_price: price at the current state
        :param next_price: price at the next state
        :return: reward from executing trade
        """
        return_rate = (next_price / current_price)
        log_return = np.log(return_rate)
        last_weight = self.current_weight.copy()
        securities_value = self.current_portfolio_value[:-1] * return_rate
        self.current_portfolio_value[:-1] = securities_value
        self.current_weight = self.current_portfolio_value / np.sum(self.current_portfolio_value)
        reward = last_weight[:-1] * log_return
        return reward

    def _init_market_data(self, data_name='market_data.pkl', pre_process=True):
        """
        load market data based on the local path and its file name, and initiate the processing and cleaning protocol of the market data
        :param data_name: file name of the pickle (serialised) file containing the market data from one or several stocks
        :param pre_process: boolean to indicate whether the data needs to be pre-processed in advance
        :return: processed market data containing expanded data, and cleaned market data without features expansion
        """
        data_path = self.data_local_path + '/' + data_name
        if not os.path.exists(data_path):
            print('market data does not exist in', self.data_local_path, '. double check, and follow instruction in ./data_wrangling directory')
        else:
            print('market data exist, loading')
            market_data = pd.read_pickle(data_path).fillna(method='ffill').fillna(method='bfill')
        if pre_process:
            processed_market_data, cleaned_market_data = StockEnv._pre_process(market_data, open_c='adj_open', close_c='adj_close', high_c='adj_high', low_c='adj_low', volume_c='adj_volume')
        assert np.sum(np.isnan(processed_market_data.values)) == 0
        assert np.sum(np.isnan(cleaned_market_data.values)) == 0
        return processed_market_data, cleaned_market_data

    def get_summary(self):
        """
        return the historical value of portfolio value, positions, and weights that the trading agent made on all states
        :return:  value of each portfolio value, positions, and weights in all states
        """
        portfolio_value_df = pd.DataFrame(np.array(self.portfolio_values), index=np.array(self.trade_dates), columns=self.instruments + ['cash'])
        positions_df = pd.DataFrame(np.array(self.positions), index=np.array(self.trade_dates), columns=self.instruments)
        weights_df = pd.DataFrame(np.array(self.weights), index=np.array(self.trade_dates), columns=self.instruments + ['cash'])
        return portfolio_value_df, positions_df, weights_df

    @staticmethod
    def _pre_process(market_data, open_c, high_c, low_c, close_c, volume_c):
        """
        make the data format consistent and check the integrity of the data
        :param market_data: file containing market data
        :param open_c: open price column name
        :param high_c: high price column name
        :param low_c: low price column name
        :param close_c: close price column name
        :param volume_c: volumn traded column name
        :return: cleaned market data having expanded features and without expanded features
        """
        preprocessed_data = {}
        cleaned_data = {}
        print(market_data.items)
        print(type(market_data))
        for c in market_data.items:
            columns = [open_c, close_c, high_c, low_c, volume_c]
            security = market_data[c, :, columns].fillna(method='ffill').fillna(method='bfill')
            security[volume_c] = security[volume_c].replace(0, np.nan).fillna(method='ffill')
            cleaned_data[c] = security.copy()
            tech_data = StockEnv._get_indicators(security=security.astype(float), open_name=open_c, close_name=close_c, high_name=high_c, low_name=low_c, volume_name=volume_c)
            preprocessed_data[c] = tech_data
        preprocessed_data = pd.Panel(preprocessed_data).dropna()
        cleaned_data = pd.Panel(cleaned_data)[:, preprocessed_data.major_axis, :].dropna()
        return preprocessed_data, cleaned_data

    @staticmethod
    def _get_indicators(security, open_name, close_name, high_name, low_name, volume_name):
        """
        expand the features of the data through technical analysis across 26 different signals
        :param security: data which features are going to be expanded
        :param open_name: open price column name
        :param close_name: close price column name
        :param high_name: high price column name
        :param low_name: low price column name
        :param volume_name: traded volumn column name
        :return: expanded and extracted data
        """
        open_price = security[open_name].values
        close_price = security[close_name].values
        low_price = security[low_name].values
        high_price = security[high_name].values
        volume = security[volume_name].values if volume_name else None
        security['MOM'] = talib.MOM(close_price)
        security['HT_DCPERIOD'] = talib.HT_DCPERIOD(close_price)
        security['HT_DCPHASE'] = talib.HT_DCPHASE(close_price)
        security['SINE'], security['LEADSINE'] = talib.HT_SINE(close_price)
        security['INPHASE'], security['QUADRATURE'] = talib.HT_PHASOR(close_price)
        security['ADXR'] = talib.ADXR(high_price, low_price, close_price)
        security['APO'] = talib.APO(close_price)
        security['AROON_UP'], _ = talib.AROON(high_price, low_price)
        security['CCI'] = talib.CCI(high_price, low_price, close_price)
        security['PLUS_DI'] = talib.PLUS_DI(high_price, low_price, close_price)
        security['PPO'] = talib.PPO(close_price)
        security['MACD'], security['MACD_SIG'], security['MACD_HIST'] = talib.MACD(close_price)
        security['CMO'] = talib.CMO(close_price)
        security['ROCP'] = talib.ROCP(close_price)
        security['FASTK'], security['FASTD'] = talib.STOCHF(high_price, low_price, close_price)
        security['TRIX'] = talib.TRIX(close_price)
        security['ULTOSC'] = talib.ULTOSC(high_price, low_price, close_price)
        security['WILLR'] = talib.WILLR(high_price, low_price, close_price)
        security['NATR'] = talib.NATR(high_price, low_price, close_price)
        security['RSI'] = talib.RSI(close_price)
        security['EMA'] = talib.EMA(close_price)
        security['SAREXT'] = talib.SAREXT(high_price, low_price)
        # security['TEMA'] = talib.EMA(close_price)
        security['RR'] = security[close_name] / security[close_name].shift(1).fillna(1)
        security['LOG_RR'] = np.log(security['RR'])
        if volume_name:
            security['MFI'] = talib.MFI(high_price, low_price, close_price, volume)
            # security['AD'] = talib.AD(high_price, low_price, close_price, volume)
            # security['OBV'] = talib.OBV(close_price, volume)
            security[volume_name] = np.log(security[volume_name])
        security.drop([open_name, close_name, high_name, low_name], axis=1)
        security = security.dropna().astype(np.float32)
        return security