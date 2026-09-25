"""Limpia dos residuos de bugs ya corregidos, para que cada celda
(config, sujeto, ratio) tenga exactamente una corrida por seed.

1. Nombres con el ratio mal redondeado: `{ratio:.1f}` escribía 0.75 como
   "ratio_0.8". El JSON siempre guardó el ratio correcto, así que el nombre es
   la única parte mal. Donde el nombre canónico ya existe para esa seed, el
   duplicado se borra; donde no, se renombra.

2. Corridas de ratio 1.0 nunca pedidas: venían de `--ratios $RATIOS "$s"`, que
   con nargs='+' se tragaba el número de sujeto y lo agregaba como un ratio
   más. Quedaron incompletas (6-7 sujetos de 8, o 39 seeds de 50), así que no
   sirven para ningún test pareado.

Los duplicados de nombre SÍ se borran: son redundantes por construcción (existe
el archivo canónico con la misma seed y la misma config). Las corridas de ratio
1.0 NO se borran, se mueven a un directorio aparte: son horas de cómputo válido,
solo que con cobertura incompleta. Quedan fuera del agregado pero recuperables
si alguna vez se completa ese punto de la curva.

Corre en seco por defecto; `--apply` ejecuta los cambios.
"""
import argparse
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TREES = ('results_ep100', 'results_008_ep100', 'results_ep100_pool2000', 'results_008_ep100_pool2000')
SPURIOUS_RATIO = 1.0
QUARANTINE = '_ratio1.0_incompleto'


def plan(root):
    """Devuelve (borrar, renombrar, apartar) sin tocar nada."""
    borrar, renombrar, apartar = [], [], []
    for tree in TREES:
        for path in glob.glob(os.path.join(root, tree, '*', 'subject_*', 'ratio_*.json')):
            with open(path) as f:
                ratio = float(json.load(f)['ratio'])
            base = os.path.basename(path)
            if ratio == SPURIOUS_RATIO:
                apartar.append(path)
                continue
            # nombre canónico = el ratio real del JSON, formateado como hoy
            seed = re.search(r'_seed_(\d+)\.json$', base).group(1)
            canon = os.path.join(os.path.dirname(path), f'ratio_{ratio}_seed_{seed}.json')
            if path == canon:
                continue
            (borrar if os.path.exists(canon) else renombrar).append(
                path if os.path.exists(canon) else (path, canon))
    return borrar, renombrar, apartar


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--root', default=HERE)
    ap.add_argument('--apply', action='store_true', help='ejecutar (por defecto: en seco)')
    args = ap.parse_args(argv)

    borrar, renombrar, apartar = plan(args.root)
    print(f'a borrar:    {len(borrar)} duplicados de nombre (el canónico existe con la misma seed)')
    print(f'a renombrar: {len(renombrar)} con el ratio redondeado y sin canónico previo')
    print(f'a apartar:   {len(apartar)} corridas de ratio {SPURIOUS_RATIO} -> {QUARANTINE}/')

    if not args.apply:
        print('\n(en seco -- usar --apply para ejecutar)')
        return
    for p in borrar:
        os.remove(p)
    for p, c in renombrar:
        os.rename(p, c)
    for p in apartar:
        dst = os.path.join(args.root, QUARANTINE, os.path.relpath(p, args.root))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.rename(p, dst)
    print(f'\nListo: {len(borrar)} borrados, {len(renombrar)} renombrados, {len(apartar)} apartados.')


def _selfcheck():
    """El plan debe borrar el duplicado cuando el canónico existe, y renombrar
    cuando no; y debe marcar el ratio espurio para borrado."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, TREES[0], 'cfg', 'subject_001')
        os.makedirs(d)

        def w(name, ratio):
            with open(os.path.join(d, name), 'w') as f:
                json.dump({'ratio': ratio, 'subject': 1}, f)

        w('ratio_0.8_seed_1.json', 0.75)    # duplicado: el canónico existe
        w('ratio_0.75_seed_1.json', 0.75)
        w('ratio_0.8_seed_2.json', 0.75)    # sin canónico: se renombra
        w('ratio_1.0_seed_3.json', 1.0)     # espurio
        w('ratio_0.5_seed_4.json', 0.5)     # sano: no se toca

        borrar, renombrar, apartar = plan(tmp)
        assert [os.path.basename(p) for p in borrar] == ['ratio_0.8_seed_1.json'], borrar
        assert [os.path.basename(p) for p, _ in renombrar] == ['ratio_0.8_seed_2.json'], renombrar
        assert os.path.basename(renombrar[0][1]) == 'ratio_0.75_seed_2.json'
        assert [os.path.basename(p) for p in apartar] == ['ratio_1.0_seed_3.json'], apartar


if __name__ == '__main__':
    _selfcheck()
    main()
