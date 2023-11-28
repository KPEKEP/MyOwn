# Copyright Pavel Nakaznenko, 2023
# For Sber beautiful code contest
# pavel@nakaznenko.com

import logging
import unittest
from decimal import Decimal
from enum import Enum
from typing import Dict, Optional

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)


class AssetPrice(Enum):
    """
    Enumeration of asset prices.
    """

    LKOH = Decimal("5896")
    SBER = Decimal("250")


class Portfolio:
    """
    A class representing a portfolio of financial assets.
    """

    def __init__(self, initial_assets: Optional[Dict[AssetPrice, int]] = None):
        """
        Initialize the Portfolio class.

        :param initial_assets: A predefined set of assets and quantities.
        """
        self.assets = initial_assets if initial_assets else {}

    def buy(self, asset: AssetPrice, quantity: int):
        """
        Purchase a specified quantity of an asset.

        :param asset: Asset to purchase.
        :param quantity: Quantity of the asset to purchase.
        :raises ValueError: If quantity is not positive.
        """
        if quantity <= 0:
            raise ValueError("Quantity should be positive.")

        self.assets[asset] = self.assets.get(asset, 0) + quantity
        logging.info(f"Bought {quantity} of {asset}. Current quantity: {self.assets[asset]}")

    def sell(self, asset: AssetPrice, quantity: int):
        """
        Sell a specified quantity of an asset.

        :param asset: Asset to sell.
        :param quantity: Quantity of the asset to sell.
        :raises ValueError: If quantity is not positive or if there aren't enough assets to sell.
        """
        if quantity <= 0:
            raise ValueError("Quantity should be positive.")
        if asset not in self.assets or self.assets[asset] < quantity:
            raise ValueError("Not enough assets to sell.")

        self.assets[asset] -= quantity
        logging.info(f"Sold {quantity} of {asset}. Remaining quantity: {self.assets[asset]}")

    def get_total_value(self) -> Decimal:
        """
        Compute the total value of the portfolio.

        :return: Total value of the portfolio.
        """
        return sum(asset.value * Decimal(quantity) for asset, quantity in self.assets.items())


class TestTradingSimulation(unittest.TestCase):
    """
    Unit tests for the trading simulation.
    """

    def setUp(self):
        """Initialize the test setup."""
        self.portfolio = Portfolio()

    def test_buy_and_sell(self):
        """Test buying and selling assets."""
        self.portfolio.buy(AssetPrice.LKOH, 5)
        self.assertEqual(self.portfolio.assets[AssetPrice.LKOH], 5)
        self.portfolio.sell(AssetPrice.LKOH, 2)
        self.assertEqual(self.portfolio.assets[AssetPrice.LKOH], 3)

    def test_get_total_value(self):
        """Test computing the total value of the portfolio."""
        self.portfolio.buy(AssetPrice.LKOH, 5)
        self.portfolio.buy(AssetPrice.SBER, 10)
        expected_value = AssetPrice.LKOH.value * Decimal(5) + AssetPrice.SBER.value * Decimal(10)
        self.assertEqual(self.portfolio.get_total_value(), expected_value)

    def test_sell_more_than_have(self):
        """Test selling more assets than are available."""
        self.portfolio.buy(AssetPrice.LKOH, 5)
        with self.assertRaises(ValueError):
            self.portfolio.sell(AssetPrice.LKOH, 6)

    def test_negative_quantity(self):
        """Test buying/selling a negative quantity."""
        with self.assertRaises(ValueError):
            self.portfolio.buy(AssetPrice.LKOH, -1)
        with self.assertRaises(ValueError):
            self.portfolio.sell(AssetPrice.LKOH, -1)

    def test_sell_asset_not_owned(self):
        """Test selling assets not owned."""
        with self.assertRaises(ValueError):
            self.portfolio.sell(AssetPrice.SBER, 5)

    def test_zero_quantity(self):
        """Test buying/selling a zero quantity."""
        with self.assertRaises(ValueError):
            self.portfolio.buy(AssetPrice.LKOH, 0)
        with self.assertRaises(ValueError):
            self.portfolio.sell(AssetPrice.LKOH, 0)

    def test_large_transactions(self):
        """Test large transactions."""
        self.portfolio.buy(AssetPrice.LKOH, 1000000)
        self.portfolio.buy(AssetPrice.SBER, 1000000)
        expected_value = AssetPrice.LKOH.value * Decimal(1000000) + AssetPrice.SBER.value * Decimal(1000000)
        self.assertEqual(self.portfolio.get_total_value(), expected_value)


if __name__ == "__main__":
    unittest.main()
