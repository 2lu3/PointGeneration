# PointGeneration
A support software for cyro electron microscope

## Setup flow

### Stable Download

Click `releases` and choose the last published version

### Latest Download

```bash
git clone https://github.com/2lu3/PointGenerationForCyro.git
```

Running adove command will create `PointGenerationForCyro` folder.
This way always downloads latest version, which may include some bugs.


### pip requirements

Running below command will install all requirements.

```bash
pip install -r requirements.txt
```

## Usage

### Input files

You can use `.txt` file for input file.
The format is shown in below. x1,y1,z1,... means decimal or integer.

```txt
    x1   y1   z1
    x2   y2   z2
    x3   y3   z3
    x4   y4   z4
```

### Output files

2 files are possibly created according to your selection in the software.
The first file contains coordinates of generated points, which format is shown in below.

```txt
    x1   y1   z1
    x2   y2   z2
    x3   y3   z3
    x4   y4   z4
```
The other file contains euler degrees, which format is shown in below.

```txt
    x1   z1   x1
    x1   z1   x1
```

