echo "1.0 / 1e-3"
python topographic_sparse_coding.py --sigma=1.0 --gamma=1e-3 --epochs=2 --verbosity=5 --dir=True

echo "0.1 / 1e-3"
python topographic_sparse_coding.py --sigma=0.1 --gamma=1e-3 --epochs=2 --verbosity=5 --dir=True

echo "10 / 1e-3"
python topographic_sparse_coding.py --sigma=10 --gamma=1e-3 --epochs=2 --verbosity=5 --dir=True

echo "1.0 / 5e-4"
python topographic_sparse_coding.py --sigma=1.0 --gamma=5e-4 --epochs=2 --verbosity=5 --dir=True

echo "1.0 / 5e-3"
python topographic_sparse_coding.py --sigma=1.0 --gamma=5e-3 --epochs=2 --verbosity=5 --dir=True
